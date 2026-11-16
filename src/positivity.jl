# Gaussian initial weights in the Hilbert–Schmidt-normalized mode frame.
# Certificates use ordinary Float64 arithmetic, with explicit PSD/trace tolerances.
using LinearAlgebra, Printf

Base.@kwdef struct PositivityOptions
    scope::String = "orbit"              # "reference" or the stronger "orbit"
    psd_tolerance::Float64 = 1e-10
    trace_tolerance::Float64 = 1e-8
    sigma::Union{Nothing,Float64} = nothing
    match_time::Union{Nothing,Float64} = nothing
    precision_accuracy::Float64 = 1e-6
    search_steps::Int = 18
    theta_grid::Int = 32
    theta_grid_max::Int = 256
    certificate_margin::Float64 = 1e-9
end
function positivity_options(config)
    d=copy(get(config,"positivity",Dict{String,Any}()))
    # Old configuration files may still specify the former filter search.
    obsolete=[k for k in ("lambda_initial","lambda_max","lambda_accuracy") if haskey(d,k)]
    if !isempty(obsolete)
        @warn "Gaussian initialization uses sigma. Ignoring legacy filter settings" obsolete
        foreach(k->delete!(d,k),obsolete)
    end
    allowed=Set(string.(fieldnames(PositivityOptions)))
    for key in keys(d)
        key in allowed || error("Unknown [positivity] option: $key")
    end
    options=PositivityOptions(;(Symbol(k)=>v for (k,v) in d)...)
    options.scope in ("reference","orbit") || error("positivity.scope must be reference or orbit")
    options.psd_tolerance>0 && options.trace_tolerance>0 || error("Positivity tolerances must be positive")
    isnothing(options.sigma) || (isfinite(options.sigma) && options.sigma>0) ||
        error("positivity.sigma must be finite and positive")
    isnothing(options.match_time) ||
        (isfinite(options.match_time) && options.match_time>=0) ||
        error("positivity.match_time must be finite and nonnegative")
    isfinite(options.precision_accuracy) && options.precision_accuracy>0 ||
        error("positivity.precision_accuracy must be finite and positive")
    options.theta_grid>=8 && options.theta_grid_max>=options.theta_grid || error("Invalid certification grid")
    options.search_steps>=1 && options.certificate_margin>0 || error("Invalid certificate search settings")
    options
end

# The columns of R are the normalized right modes. In this frame eta=R*R'.
# The polar factor eta^(-1/2)*R maps coordinate vectors to the effective kets.
# Evolved overlap phases set theta=0 at the Gaussian start.
function gaussian_basis(b,N;phase_time=0.)
    sizes=[norm(view(b.R,:,j)) for j in eachindex(b.ell)]
    all(x->isfinite(x) && x>0,sizes) || error("Invalid right-mode norm")
    R=b.R ./ reshape(sizes,1,:)
    j0=findfirst(==(0),b.ell)
    j0===nothing && error("Stationary harmonic l=0 is required")
    stationary_trace=tr(reshape(view(R,:,j0),N,N))
    abs(stationary_trace)>0 || error("Stationary right mode has zero trace")
    scale=1/abs(stationary_trace)
    c=zeros(ComplexF64,length(b.ell))
    c[j0]=inv(stationary_trace)
    for j in findall(>(0),b.ell)
        jm=findfirst(==(-b.ell[j]),b.ell)
        jm===nothing && error("Missing conjugate harmonic")
        anchor=b.c[j]*cis(phase_time*imag(b.Z[j]))
        if iszero(anchor)
            # This harmonic has no phase in the coherent projection.
            mode=view(R,:,j)
            pivot=mode[argmax(abs.(mode))]
            c[j]=scale*conj(pivot)/abs(pivot)
        else
            c[j]=scale*anchor/abs(anchor)
        end
        partner=vec(adjoint(reshape(view(R,:,j),N,N)))
        overlap=dot(view(R,:,jm),partner)
        abs(overlap)>0 || error("Conjugate harmonics have incompatible right modes")
        c[jm]=conj(c[j])*overlap/abs(overlap)
    end
    merge(b,(;R,c))
end

# precision = 1/(4*sigma^2), so squared amplitudes are exp(-l^2/(2*sigma^2)).
function gaussian_weights(b,precision,t)
    logs=-2precision .* b.ell.^2 .+ 2t .* real.(b.Z)
    logs .-= maximum(logs)
    w=exp.(logs)
    w./sum(w)
end

function gaussian_pair(b,N,precision,t,theta)
    amp=b.c .* exp.(-precision .* b.ell.^2 .+ b.Z.*t .+ im.*b.ell.*theta)
    reshape(b.R*amp,N,N),reshape(b.R*(im.*b.ell.*amp),N,N)
end

function projected_weights(raw,tau)
    # Normalize after decay and keep small weights in logarithmic form.
    logs=[2log(abs(raw.c[j]))+log(sum(abs2,view(raw.R,:,j)))+2tau*real(raw.Z[j])
          for j in eachindex(raw.ell)]
    peak=maximum(logs)
    isfinite(peak) || error("Evolved coherent projection has zero norm")
    w=exp.(logs.-peak)
    w./sum(w)
end

function match_precision(b,target)
    target>0 || error("Projected variance at match_time is numerically zero. Choose an earlier match_time or set sigma.")
    l2=b.ell.^2
    # Paired Gaussian weights have zero mean at all times.
    variance(beta)=dot(gaussian_weights(b,beta,0.),l2)
    broadest=variance(0.)
    if target>=broadest
        if target>broadest
            @warn "Projected variance exceeds the broadest fresh centered Gaussian" target broadest
        end
        return 0.
    end
    lo=0.;hi=1.
    while variance(hi)>target
        hi*=2
    end
    for _ in 1:50
        mid=(lo+hi)/2
        if variance(mid)>target
            lo=mid
        else
            hi=mid
        end
    end
    hi
end

function prepare_certificate(b,N,o)
    length(unique(b.ell))==length(b.ell) || error("Duplicate harmonic labels")
    j0=findfirst(==(0),b.ell)
    j0===nothing && error("Stationary harmonic l=0 is required")
    abs(b.Z[j0])<1e-10 || error("l=0 must be stationary")
    stationary=reshape(b.R[:,j0]*b.c[j0],N,N)
    norm(stationary-stationary')<=o.trace_tolerance || error("Stationary component is not Hermitian")
    abs(tr(stationary)-1)<=o.trace_tolerance || error("Stationary component must have unit trace")
    stationary_trace_error=abs(tr(stationary)-1)
    stationary=Matrix(Hermitian((stationary+stationary')/2))
    E=eigen(Hermitian(stationary))
    m0=minimum(E.values)
    # Adding tolerance*I is only a proof device. It is NOT added to the output state.
    minimum(E.values .+ o.psd_tolerance)>0 || error(
        "Stationary component is not numerically PSD. Gaussian filtering cannot repair it; improve eigensolver accuracy.")
    S=E.vectors*Diagonal(1 ./ sqrt.(E.values .+ o.psd_tolerance))
    positive=findall(>(0),b.ell)
    length(b.ell)==2length(positive)+1 || error("Need stationary mode and conjugate harmonic pairs")
    C=Matrix{ComplexF64}[]; norms=Float64[]; trace_bounds=Float64[]; pairing_bounds=Float64[]
    for j in positive
        l=b.ell[j]; jm=findfirst(==(-l),b.ell)
        jm===nothing && error("Missing conjugate harmonic -$l")
        v=reshape(b.R[:,j]*b.c[j],N,N)
        vm=reshape(b.R[:,jm]*b.c[jm],N,N)
        norm(vm-v')<=o.trace_tolerance*max(1.,norm(v)) || error("Harmonic $l violates Hermitian conjugacy")
        b.Z[jm]==conj(b.Z[j]) || error("Spectrum violates conjugate pairing")
        real(b.Z[j])<=0 || error("Growing mode at l=$l is incompatible with the slow dissipative sector")
        push!(trace_bounds,abs(tr(v))+abs(tr(vm)))
        discrepancy=vm-v'
        push!(pairing_bounds,iszero(discrepancy) ? 0. : norm(S'*discrepancy*S))
        cc=S'*v*S
        push!(C,cc);push!(norms,norm(cc)) # Frobenius norm bounds the operator norm.
    end
    (;stationary_minimum=m0,stationary_trace_error,positive,C,norms,trace_bounds,pairing_bounds,N)
end

function certify_gaussian(b,prepared,o,precision,times)
    indices=prepared.positive
    ls=Float64.(b.ell[indices]); cf=exp.(-precision.*ls.^2)
    trace_error=prepared.stationary_trace_error+sum(cf.*prepared.trace_bounds)
    trace_error<=o.trace_tolerance || return (;ok=false,minimum_bound=-Inf,grid=0,trace_error)
    best=-Inf;used=0
    # Congruence W=(rho_ss+tol*I)^(-1/2) rho (rho_ss+tol*I)^(-1/2).
    # Certify I + sum(C_l e^{il theta}+h.c.) >= 0, hence rho >= -tol*I.
    for t in times
        amplitudes=exp.(-precision.*ls.^2 .+ b.Z[indices].*t)
        sizes=abs.(amplitudes).*prepared.norms
        pairing_error=dot(abs.(amplitudes),prepared.pairing_bounds)
        direct=1-2sum(sizes)-pairing_error
        if direct>=o.certificate_margin
            best=min(isfinite(best) ? best : Inf,direct)
            continue
        end
        if o.scope=="reference"
            H=Matrix{ComplexF64}(I,prepared.N,prepared.N)
            for k in eachindex(indices)
                term=amplitudes[k]*prepared.C[k]
                H .+= term .+ term'
            end
            lower=eigmin(Hermitian(H))-pairing_error
            lower>=o.certificate_margin || return (;ok=false,minimum_bound=lower,grid=1,trace_error)
            best=min(isfinite(best) ? best : Inf,lower);used=max(used,1)
            continue
        end
        # Between samples the Hermitian matrix differs from its linear interpolant
        # by at most h^2/8 * sup ||d²W/dtheta²||. Weyl's bound certifies ALL angles.
        curvature=2sum(ls.^2 .* sizes)
        M=o.theta_grid;accepted=false;lower=-Inf
        while M<=o.theta_grid_max
            sampled=Inf
            for j in 0:M-1
                theta=2pi*j/M
                H=Matrix{ComplexF64}(I,prepared.N,prepared.N)
                for k in eachindex(indices)
                    term=amplitudes[k]*cis(ls[k]*theta)*prepared.C[k]
                    H .+= term .+ term'
                end
                sampled=min(sampled,eigmin(Hermitian(H)))
                # A negative sample cannot be rescued by refining the angle grid.
                sampled<o.certificate_margin && break
            end
            lower=sampled-(2pi/M)^2*curvature/8-pairing_error
            if lower>=o.certificate_margin
                accepted=true;used=max(used,M);break
            end
            sampled<o.certificate_margin && break
            M*=2
        end
        accepted || return (;ok=false,minimum_bound=lower,grid=min(M,o.theta_grid_max),trace_error)
        best=min(isfinite(best) ? best : Inf,lower)
    end
    (;ok=true,minimum_bound=best,grid=used,trace_error)
end

function gaussian_match_time(o,times)
    isnothing(o.match_time) || return o.match_time
    first_positive=minimum(filter(>(0),times);init=Inf)
    isfinite(first_positive) ? first_positive : 0.
end

function restore_reference(raw,b,N,o,times)
    checktimes=sort(unique(vcat(0.,Float64.(times))))
    tau=gaussian_match_time(o,checktimes)
    prepared=prepare_certificate(b,N,o)
    evolved_projection=projected_weights(raw,tau)
    target=fs_metric(evolved_projection,b.ell)
    initial=isnothing(o.sigma) ? match_precision(b,target) : inv(4*o.sigma^2)
    isfinite(initial) && initial>=0 || error("Gaussian width is outside floating-point range")
    hi=initial
    cert=certify_gaussian(b,prepared,o,hi,checktimes)
    if !cert.ok
        lo=hi
        hi=max(2hi,1e-4)
        while true
            cert=certify_gaussian(b,prepared,o,hi,checktimes)
            cert.ok && break
            hi<1024 || error("Could not certify a Gaussian initial state. Check the stationary state and angular resolution.")
            lo=hi
            hi=min(2hi,1024.)
        end
        for _ in 1:o.search_steps
            hi-lo<=o.precision_accuracy && break
            mid=(lo+hi)/2
            trial=certify_gaussian(b,prepared,o,mid,checktimes)
            if trial.ok
                hi=mid
                cert=trial
            else
                lo=mid
            end
        end
    end
    sigma=iszero(hi) ? Inf : inv(2sqrt(hi))
    p=gaussian_weights(b,hi,0.)
    sum(p[b.ell .!= 0])>0 || error("Certified Gaussian width has no resolved orbital harmonics")
    @printf("Gaussian initial width sigma=%.10g, continuum certificate margin %.3e\n",sigma,cert.minimum_bound)
    trial_variance=fs_metric(gaussian_weights(b,initial,0.),b.ell)
    selected_variance=fs_metric(p,b.ell)
    initial_weight_distance=sum(abs,p.-projected_weights(raw,0.))/2
    matched_weight_distance=sum(abs,p.-evolved_projection)/2
    if hi>initial
        @warn "Positivity required a narrower Gaussian. See the variance at match_time in positivity_report.txt." tau target selected_variance
    end
    (;sigma,precision=hi,initial_sigma=iszero(initial) ? Inf : inv(2sqrt(initial)),
      match_time=tau,projected_variance=target,trial_variance,selected_variance,
      initial_weight_distance,matched_weight_distance,
      certificate=cert,stationary_minimum=prepared.stationary_minimum,times=checktimes)
end

function export_reference(raw,b,q,o,result,folder;seed)
    precision=result.precision
    rho,_=gaussian_pair(b,q.N,precision,0.,0.)
    m=eigmin(Hermitian((rho+rho')/2))
    raw_rho=reshape(raw.R*raw.c,q.N,q.N)
    raw_minimum=eigmin(Hermitian((raw_rho+raw_rho')/2))
    evolved_raw=reshape(raw.R*(raw.c.*exp.(raw.Z.*result.match_time)),q.N,q.N)
    evolved_raw_minimum=eigmin(Hermitian((evolved_raw+evolved_raw')/2))
    m>=-o.psd_tolerance || error("Gaussian reference failed final PSD check")
    abs(tr(rho)-1)<=o.trace_tolerance || error("Gaussian reference failed trace check")
    norm(rho-rho')<=o.trace_tolerance || error("Gaussian reference failed Hermiticity check")
    writedlm(joinpath(folder,"reference_state_real.csv"),real.(rho),',')
    writedlm(joinpath(folder,"reference_state_imag.csv"),imag.(rho),',')
    p=gaussian_weights(b,precision,0.)
    c=b.c .* exp.(-precision .* b.ell.^2)
    write_csv(joinpath(folder,"reference_harmonics.csv"),
        ["l","gaussian_weight","log_gaussian_envelope","raw_c_real","raw_c_imag","gaussian_c_real","gaussian_c_imag"],
        hcat(b.ell,p,-2precision.*b.ell.^2,real.(raw.c),imag.(raw.c),real.(c),imag.(c)))
    open(joinpath(folder,"positivity_report.txt"),"w") do io
        println(io,"method=Gaussian initial weights fitted at the shifted time origin")
        println(io,"sigma=$(result.sigma)")
        println(io,"initial_sigma=$(result.initial_sigma)")
        println(io,"match_time=$(result.match_time)")
        println(io,"gaussian_start_projection_time=$(result.match_time)")
        println(io,"certified_times_origin=Gaussian start")
        println(io,"coherent_seed_t_star=$(seed.t_star)")
        println(io,"coherent_seed_x=$(seed.x0)")
        println(io,"coherent_seed_p=$(seed.p0)")
        println(io,"projected_variance_at_match_time=$(result.projected_variance)")
        println(io,"trial_gaussian_variance_at_match_time=$(result.trial_variance)")
        println(io,"selected_gaussian_variance_at_match_time=$(result.selected_variance)")
        println(io,"weight_total_variation_at_zero=$(result.initial_weight_distance)")
        println(io,"weight_total_variation_at_match_time=$(result.matched_weight_distance)")
        println(io,"gaussian_precision=$precision")
        println(io,"initial_phase_variance=$(fs_metric(p,b.ell))")
        println(io,"eta=sum of Hilbert-Schmidt-normalized right-mode projectors")
        println(io,"phase_origin=evolved coherent overlap phases at match_time when nonzero")
        println(io,"scope=$(o.scope)")
        println(io,"certified_times=$(join(result.times,','))")
        println(io,"psd_tolerance=$(o.psd_tolerance)")
        println(io,"trace_tolerance=$(o.trace_tolerance)")
        println(io,"stationary_minimum_eigenvalue=$(result.stationary_minimum)")
        println(io,"raw_reference_minimum_eigenvalue=$raw_minimum")
        println(io,"evolved_projection_minimum_eigenvalue_at_match_time=$evolved_raw_minimum")
        println(io,"reference_minimum_eigenvalue=$m")
        println(io,"reference_trace_real=$(real(tr(rho)))")
        println(io,"reference_hermiticity_error=$(norm(rho-rho'))")
        println(io,"relative_continuum_certificate_margin=$(result.certificate.minimum_bound)")
        println(io,"certification_angle_grid=$(result.certificate.grid)")
        println(io,"trace_error_bound=$(result.certificate.trace_error)")
        println(io,"all_theta_certified=$(o.scope=="orbit")")
        println(io,"Certificate uses Float64 and the bound rho(theta,t) >= -psd_tolerance*I.")
    end
end
