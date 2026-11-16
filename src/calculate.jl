#!/usr/bin/env julia
using TOML, SHA, Serialization, Random
include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "phase.jl"))
include(joinpath(@__DIR__, "positivity.jl"))
include(joinpath(@__DIR__, "positive_geometry.jl"))

# Serialized caches are local Julia data: load only caches you trust.
# Source hashes invalidate numerical results after implementation changes.
function fingerprint(parts...)
    bytes2hex(sha256(join(string.(parts), "\n")))
end
function atomic_serialize(path, value)
    tmp=path*".tmp"
    open(tmp,"w") do io
        serialize(io,value)
    end
    mv(tmp,path;force=true)
end
function cached(f::Function, folder, label, key)
    path=joinpath(folder,label*"_"*key*".jls")
    if isfile(path)
        try
            value=deserialize(path)
            println("REUSE ",label," ",key[1:12])
            return value
        catch err
            @warn "Unreadable cache; rebuilding" path exception=err
        end
    end
    println("CALCULATE ",label," ",key[1:12])
    value=f()
    atomic_serialize(path,value)
    value
end
function export_spectrum(b,folder)
    write_csv(joinpath(folder,"spectrum.csv"),["l","Re_z","Im_z","gamma"],
              hcat(b.ell,real.(b.Z),imag.(b.Z),-real.(b.Z)))
end
function main_calculate()
    length(ARGS)==2 || error("Usage: julia --project=. src/calculate.jl CONFIG.toml WORKDIR")
    config=TOML.parsefile(ARGS[1]); work=abspath(ARGS[2])
    model=config["model"]; numerics=config["numerics"]; obs=config["observations"]
    folder=joinpath(work,"data"); cache=joinpath(work,"cache")
    mkpath(folder);mkpath(cache)
    q=Params(;N=Int(numerics["N"]),K=Int(numerics["K"]),
        Delta=Float64(model["omega0"]),epsilon=Float64(model["epsilon"]),
        gamma_up=Float64(model["gamma_up"]),gamma_down=Float64(model["gamma_down"]),
        nu=Float64(model["kappa"]),times=sort(unique(vcat(Float64.(obs["times"]),Float64(obs["bures_time"])))),
        ngrid=Int(numerics["husimi_grid"]),nangle=Int(numerics["bures_angles"]),
        nev=Int(numerics["nev"]),tol=Float64(numerics["tol"]),
        maxiter=Int(numerics["maxiter"]))
    q.N>=8 && q.K>=5 || error("Require N>=8, K>=5")
    q.Delta>abs(q.epsilon) || error("Require omega0 > |epsilon|")
    q.gamma_up>q.gamma_down && q.nu>0 || error("Require pumping > damping and kappa > 0")
    q.ngrid>=8 && q.nangle>=8 || error("Grid sizes must be >=8")
    !isempty(q.times) && all(isfinite,q.times) && minimum(q.times)>=0 || error("Invalid observation times")
    bt=Float64(obs["bures_time"])
    bt>=0 && isfinite(bt) || error("Invalid Bures observation time")
    harmonics=sort(unique(Int.(obs["phase_harmonics"])))
    !isempty(harmonics) && minimum(harmonics)>=1 && maximum(harmonics)<=q.K || error("Phase harmonics must lie in 1:K")
    # Stable physical/solver key: times, sampling grids and phase selection are excluded.
    source=read(joinpath(@__DIR__,"model.jl"),String)
    key=fingerprint(source,VERSION,q.N,q.K,q.Delta,q.epsilon,q.gamma_up,
                    q.gamma_down,q.nu,q.nev,q.tol,q.maxiter,numerics["seed"])
    Random.seed!(Int(numerics["seed"]))
    period,omega=estimate_period(q)
    cyc=export_classical(q,period,omega,folder)
    seed_index=argmax(abs2.(cyc[:,2]).+abs2.(cyc[:,3]))
    t_star,x0,p0=cyc[seed_index,1:3]
    alpha=(x0+im*p0)/sqrt(2)
    psi=coherent(q.N,alpha)
    @printf("Maximum-radius coherent seed: t*=%.9g, x0=%.9g, p0=%.9g, missing norm=%.3e\n",
            t_star,x0,p0,max(0.,1-sum(abs2,psi)))
    norm(psi)>1e-12 || error("Coherent reference outside Fock cutoff")
    psi./=norm(psi)
    # A single basis supplies stationary state, spectral geometry and phase data.
    b=cached(cache,"basis",key) do
        L=get_liouvillian(q)
        even,_=parity_indices(q.N)
        rho=stationary(L[even,even],even,q.N,q)
        ell,Z,R,c,U=slow_modes(L,q,psi*psi',rho,omega,folder)
        (;rho,ell,Z,R,c,U)
    end
    # Reproject using the cached biorthogonal left modes.
    overlaps=b.U'*vec(psi*psi')
    b=merge(b,(c=vcat(reverse(conj.(overlaps)),norm(b.rho),overlaps),))
    export_spectrum(b,folder)
    options=positivity_options(config)
    tau=gaussian_match_time(options,q.times)
    gaussian=gaussian_basis(b,q.N;phase_time=tau)
    positivity_source=read(joinpath(@__DIR__,"positivity.jl"),String)
    positivity_key=fingerprint(key,positivity_source,options,q.times,bt,t_star,x0,p0)
    restoration=cached(cache,"gaussian_reference",positivity_key) do
        restore_reference(b,gaussian,q.N,options,sort(unique(vcat(0.,q.times,bt))))
    end
    export_reference(b,gaussian,q,options,restoration,folder;seed=(;t_star,x0,p0))
    function export_cached(label, stagekey, names, producer)
        payload=cached(cache,label,stagekey) do
            producer()
            Dict(name=>read(joinpath(folder,name)) for name in names)
        end
        for (name,bytes) in payload
            write(joinpath(folder,name),bytes)
        end
    end
    export_cached("husimi",fingerprint(key,q.ngrid),["husimi_q.csv","photon_pop.csv"],
                  ()->husimi(q,b.rho,cyc,folder))
    geomfiles=["weights.csv","fs_selected.csv","fidelity.csv","metric_time.csv",
               "bures_absolute.csv","bures_repaired_mask.csv","metadata.txt",
               "bures_min_eigenvalue.csv","bures_trace_error.csv","bures_hermiticity_error.csv"]
    geometry_source=read(joinpath(@__DIR__,"positive_geometry.jl"),String)
    export_cached("positive_geometry",fingerprint(positivity_key,geometry_source,q.nangle),geomfiles,
        ()->export_positive_geometry(q,gaussian,omega,folder,options,restoration;bures_time=bt))
    phasefiles=["theta_comparison.csv","harmonic_amplitudes.csv","summary.txt","comparison_parameters.txt"]
    phasekey=fingerprint(key,read(joinpath(@__DIR__,"phase.jl"),String),harmonics,obs["phase_points"])
    export_cached("phase",phasekey,phasefiles,
        ()->export_phase(q,b,harmonics,folder,Int(obs["phase_points"])))
    cp(ARGS[1],joinpath(work,"config_used.toml");force=true)
    println("Data ready: ",folder)
end
main_calculate()
