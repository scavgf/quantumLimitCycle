# Shared numerical model for all five manuscript figures.
# H = Delta*a†a + epsilon/2*(a²+a†²)
# Dissipation: gamma_up*D[a†] + gamma_down*D[a] + nu*D[a²].
# Slow right modes are HS normalized; U stores their biorthogonal left modes.
using LinearAlgebra, SparseArrays, DelimitedFiles, Printf, Statistics, Arpack

Base.@kwdef struct Params
    N::Int=64; K::Int=8
    Delta::Float64=1.0; epsilon::Float64=0.30
    gamma_up::Float64=1.0; gamma_down::Float64=0.2; nu::Float64=0.0125
    outdir::String="results"
    times::Vector{Float64}=[0.,2.,5.,10.,15.,20.]
    ngrid::Int=101; nangle::Int=48
    nev::Int=8; tol::Float64=1e-8; maxiter::Int=4000
end
function ann(N)
    sparse(collect(1:N-1),collect(2:N),ComplexF64.(sqrt.(1:N-1)),N,N)
end
function dissipator(J,I0)
    M=sparse(J'*J)
    kron(conj(J),J)-0.5*(kron(I0,M)+kron(transpose(M),I0))
end
function get_liouvillian(q)
    a=ann(q.N); ad=sparse(a'); I0=spdiagm(0=>ones(ComplexF64,q.N))
    H=sparse(q.Delta*(ad*a)+(q.epsilon/2)*(a*a+ad*ad))
    L=-im*(kron(I0,H)-kron(transpose(H),I0))
    L+=q.gamma_up*dissipator(ad,I0)
    L+=q.gamma_down*dissipator(a,I0)
    L+=q.nu*dissipator(a*a,I0)
    dropzeros!(L)
    return L
end
# For this model, simultaneous parity of ket and bra is preserved by L.
function parity_indices(N)
    even=Int[];odd=Int[]
    for j in 1:N, i in 1:N
        ix=i+(j-1)*N
        push!(iseven(i-j) ? even : odd,ix)
    end
    even,odd
end
function meanfield(x,p,q)
    # alpha=(x+i*p)/sqrt(2), dot(alpha)=(mu-iDelta)alpha
    #                      -i epsilon alpha* -nu |alpha|^2 alpha
    mu=(q.gamma_up-q.gamma_down)/2-q.nu*(x*x+p*p)/2
    return mu*x+(q.Delta-q.epsilon)*p,
           mu*p-(q.Delta+q.epsilon)*x
end
function rk4(x,p,h,q)
    k1=meanfield(x,p,q)
    k2=meanfield(x+h*k1[1]/2,p+h*k1[2]/2,q)
    k3=meanfield(x+h*k2[1]/2,p+h*k2[2]/2,q)
    k4=meanfield(x+h*k3[1],p+h*k3[2],q)
    return x+h*(k1[1]+2k2[1]+2k3[1]+k4[1])/6,
           p+h*(k1[2]+2k2[2]+2k3[2]+k4[2])/6
end
function estimate_period(q;dt=0.01)
    # Use two consecutive positive-x, decreasing-phase crossings p=0.
    abs(q.Delta)>abs(q.epsilon) || error("Example oscillator requires |Delta|>|epsilon| for circulating phase")
    r=sqrt(max(1.,2*(q.gamma_up-q.gamma_down)/q.nu))
    x,p=r,0.; crossings=Float64[]
    Tburn=120.; Tobs=65.
    for k in 1:ceil(Int,(Tburn+Tobs)/dt)
        xp,pp=x,p
        x,p=rk4(x,p,dt,q)
        t=k*dt
        if t>Tburn && pp>0 && p<=0 && x>0
            frac=pp/(pp-p)
            push!(crossings,t-dt+frac*dt)
        end
    end
    length(crossings)>=3 || error("Could not identify stable rotating classical cycle. Check parameters.")
    period=mean(diff(crossings[end-2:end]))
    omega=2pi/period
    return period,omega
end
function write_csv(path,headers,values)
    open(path,"w") do io
        println(io,join(headers,","))
        writedlm(io,values,',')
    end
end
function export_classical(q,period,omega,folder)
    dt=min(0.015,period/400)
    seeds=[(-12.,5.),(11.,-8.),(2.,2.),(-3.,9.)]
    approach=Float64[]
    for (j,(x0,p0)) in enumerate(seeds)
        x,p=x0,p0
        for k in 0:ceil(Int,3*period/dt)
            append!(approach,[j-1.,k*dt,x,p])
            x,p=rk4(x,p,dt,q)
        end
    end
    writedlm(joinpath(folder,"approach.csv"),reshape(approach,4,:)',',')
    # Relax and align to positive x crossing p=0.
    x,p=12.,0.
    for _ in 1:ceil(Int,150/dt)
        x,p=rk4(x,p,dt,q)
    end
    for _ in 1:ceil(Int,2*period/dt)
        xx,pp=x,p
        x,p=rk4(x,p,dt,q)
        if pp>0 && p<=0 && x>0
            break
        end
    end
    m=601; cyc=zeros(m,4)
    h=period/(m-1)
    for k in 1:m
        vx,vp=meanfield(x,p,q)
        cyc[k,:]=[(k-1)*h,x,p,1/(period*max(hypot(vx,vp),1e-10))]
        if k<m
            x,p=rk4(x,p,h,q)
        end
    end
    writedlm(joinpath(folder,"cycle.csv"),cyc,',')
    return cyc
end
function candidates(A,sigma,q)
    vals,vecs,nconv,_,_,_=eigs(A;nev=min(q.nev,size(A,1)-2),
          ncv=min(size(A,1),max(25,2*q.nev+5)),sigma=sigma,
          which=:LM,tol=q.tol,maxiter=q.maxiter)
    nconv>0 || error("ARPACK no convergence at sigma=$sigma")
    return vals[1:nconv],vecs[:,1:nconv]
end
function stationary(A,even,N,q)
    vals,V=candidates(A,-0.00013+0im,q)
    j=argmin(abs.(vals)); z=vals[j]
    abs(z)<1e-5 || @warn "Steady eigenvalue not nearly zero" z
    full=zeros(ComplexF64,N*N);full[even]=V[:,j]
    rho=reshape(full,N,N)
    # An eigensolver returns an arbitrary complex phase. Normalize the trace
    # BEFORE taking the Hermitian part, or an imaginary phase can annihilate it.
    trraw=tr(rho)
    abs(trraw)>1e-12 || error("Candidate stationary mode has nearly zero trace")
    rho=Matrix(Hermitian((rho/trraw+(rho/trraw)')/2))
    rho ./=real(tr(rho))
    lambda=eigvals(Hermitian(rho))
    minimum(lambda)<-1e-5 && @warn "Steady state not PSD: check solver/cutoff" minimum(lambda)
    return rho
end
function coherent(N,alpha)
    v=zeros(ComplexF64,N);v[1]=exp(-abs2(alpha)/2)
    for i in 1:N-1
        v[i+1]=v[i]*alpha/sqrt(i)
    end
    # Do NOT renormalize a truncated coherent vector: grid points above the
    # Fock cutoff would otherwise acquire spurious Husimi Q intensity.
    v
end
function husimi(q,rho,cyc,folder)
    extent=1.25*maximum(abs.(cyc[:,2:3]))
    extent=max(extent,7.)
    xs=collect(range(-extent,extent,length=q.ngrid));ps=xs
    rows=zeros(length(xs)*length(ps),3);k=0
    for p in ps, x in xs
        k+=1
        v=coherent(q.N,(x+im*p)/sqrt(2))
        rows[k,:]=[x,p,max(0.,real(dot(v,rho*v)))/pi]
    end
    writedlm(joinpath(folder,"husimi_q.csv"),rows,',')
    pop=real.(diag(rho))
    writedlm(joinpath(folder,"photon_pop.csv"),hcat(0:q.N-1,pop),',')
    upper=sum(abs.(pop[max(1,q.N-9):end]))
    @printf("Fock top-10 population = %.3e (must converge with N)\n",upper)
    return extent
end
function select_mode(A, ell, omega, q, diffusion_a)
    # With z_l = -gamma_l - i*epsilon_l and gamma_l ≈ a*l^2,
    # the shift must include the quadratic damping of high harmonics.
    gamma_est = diffusion_a * ell^2
    shift = -gamma_est - im*ell*omega
    vals, V = candidates(A, shift, q)
    err_freq = abs.(imag.(vals) .+ ell*omega)
    # Frequency tolerance prevents matching a neighboring harmonic.
    admissible = findall(j -> real(vals[j]) < 1e-6 &&
                             err_freq[j] < 0.4*abs(omega), eachindex(vals))
    if isempty(admissible)
        @warn "No harmonic near estimated shift" ell shift diffusion_a
        for z in vals
            @printf("  candidate z=%12.6f%+12.6fi  Δω=%.5g\n",
                    real(z), imag(z), abs(imag(z)+ell*omega))
        end
        error("No unambiguous slow harmonic l=$ell near $shift. Consider lowering K, checking parameters, or inspecting the slow spectrum")
    end
    # Match the predicted complex eigenvalue rather than blindly preferring
    # the smallest damping (which biases selection toward unrelated branches).
    j = admissible[argmin([abs(vals[k] - shift) for k in admissible])]
    z = vals[j]
    r = V[:,j] / norm(V[:,j])
    residual = norm(A*r - z*r) / max(norm(r), 1e-15)
    residual < 1e-4 || @warn "Large mode residual" ell residual
    return z, r
end
function left_mode(A,z,r,q)
    vals,V=candidates(sparse(adjoint(A)),conj(z)-0.005,q)
    j=argmin(abs.(vals.-conj(z)))
    abs(vals[j]-conj(z))<1e-4 || error("Left mode mismatch for $z")
    u=V[:,j];overlap=dot(u,r)
    abs(overlap)>1e-10 || error("Poorly conditioned biorthogonal mode; overlap=$overlap")
    u ./=conj(overlap)  # dot(u,r) = 1
    return u
end
function slow_modes(L,q,rho0,rhosteady,omega,folder)
    even,odd=parity_indices(q.N)
    nnz(L[even,odd])+nnz(L[odd,even])==0 || error("Parity block decomposition invalid")
    blocks=(L[even,even],L[odd,odd]);indices=(even,odd)
    ell=collect(-q.K:q.K); Z=zeros(ComplexF64,length(ell))
    coeff=zeros(ComplexF64,length(ell))
    R=zeros(ComplexF64,q.N*q.N,length(ell))
    j0=q.K+1;ns=norm(rhosteady)
    R[:,j0].=vec(rhosteady)./ns; coeff[j0]=ns;Z[j0]=0
    U=zeros(ComplexF64,q.N*q.N,q.K)
    ref=vec(rho0)
    # Start with a small positive damping coefficient; update it by fitting
    # the already identified low harmonics, gamma_l ≈ diffusion_a*l^2.
    fitted_l = Float64[]
    fitted_gamma = Float64[]
    diffusion_a = 0.005
    for l in 1:q.K
        parity=iseven(l) ? 1 : 2
        A=blocks[parity];idx=indices[parity]
        z,r=select_mode(A,l,omega,q,diffusion_a)
        u=left_mode(A,z,r,q)
        U[idx,l].=u
        jplus=j0+l;jminus=j0-l
        R[idx,jplus].=r
        R[:,jminus].=vec(adjoint(reshape(R[:,jplus],q.N,q.N)))
        coeff[jplus]=dot(u,ref[idx]);coeff[jminus]=conj(coeff[jplus])
        Z[jplus]=z;Z[jminus]=conj(z)
        # Fit only l <= 5 to avoid distortions at high harmonics.
        if l <= 5
            push!(fitted_l, Float64(l))
            push!(fitted_gamma, -real(z))
            xx = fitted_l .^ 2
            diffusion_a = dot(xx, fitted_gamma) / dot(xx, xx)
            @printf("  Updated high-harmonic shift: Re(z_l) ≈ -%.7g*l^2\n", diffusion_a)
        end
        @printf("l=%3d   gamma=%11.6g   frequency=%11.6g   |c|=%10.4g\n",
                l,-real(z),-imag(z),abs(coeff[jplus]))
    end
    writedlm(joinpath(folder,"spectrum.csv"),
        vcat(reshape(["l","Re_z","Im_z","gamma"],1,4),
             Matrix{Any}(hcat(ell,real.(Z),imag.(Z),-real.(Z)))),',')
    return ell,Z,R,coeff,U
end
function weights(coeff,z,t)
    logs=2 .* (log.(max.(abs.(coeff),1e-250)).+real.(z).*t)
    logs .-= maximum(logs)
    a=exp.(logs)
    a./sum(a)
end
function fs_metric(p,ell)
    m=sum(p.*ell)
    return sum(p.*(ell.-m).^2)
end
function save_wide(path,header,grid,times,data)
    labels=[header;[@sprintf("t_%g",t) for t in times]]
    write_csv(path,labels,hcat(grid,data))
end
# Bures surrogate. This is not physical QFI when reconstructed operator is not PSD.
# If projection has negative eigenvalues, clip rho for the diagnostic only;
# derivative remains that of unprojected slow manifold, as in earlier pipeline.
function bures_surrogate(rho,drho)
    E=eigen(Hermitian((rho+rho')/2)); lam=E.values
    wasrepaired=minimum(lam)<-1e-8
    lam=max.(lam,0.)
    lam ./=max(sum(lam),1e-15)
    A=E.vectors'*drho*E.vectors
    g=0.
    for j in eachindex(lam),i in eachindex(lam)
        d=lam[i]+lam[j]
        d>1e-9 && (g+=abs2(A[i,j])/(2d))
    end
    return g,wasrepaired
end
function export_geometry(q,ell,Z,R,c,omega,folder; bures_time=4.0)
    times=sort(unique(q.times)); T=length(times)
    length(times)>0 || error("Specify at least one observation time")
    W=hcat([weights(c,Z,t) for t in times]...)
    FS=[fs_metric(W[:,j],ell) for j in 1:T]
    save_wide(joinpath(folder,"weights.csv"),"l",ell,times,W)
    write_csv(joinpath(folder,"fs_selected.csv"),["t","g_FS"],hcat(times,FS))
    delta=collect(range(-pi,pi,length=301))
    fidelity=zeros(length(delta),T)
    for (j,p) in enumerate(eachcol(W)),(i,d) in enumerate(delta)
        fidelity[i,j]=abs2(sum(p.*exp.(im.*ell.*d)))
    end
    save_wide(joinpath(folder,"fidelity.csv"),"delta_theta_over_pi",delta./pi,times,fidelity)
    tg=sort(unique(vcat(collect(range(0.,max(20.,maximum(times)),length=201)),times)))
    metric=[fs_metric(weights(c,Z,t),ell) for t in tg]
    write_csv(joinpath(folder,"metric_time.csv"),["t","g_FS"],hcat(tg,metric))
    # Only Fig. 1 needs this costly angular reconstruction.
    # Its time is independent of the Fig. 2 selected times.
    btimes=[bures_time]
    theta=collect(0:q.nangle-1).*2pi/q.nangle
    b=zeros(q.nangle,length(btimes));repair=zeros(q.nangle,length(btimes))
    for (j,t) in enumerate(btimes), (i,th) in enumerate(theta)
        v=c.*exp.(Z.*t).*(exp.(im.*ell.*th))
        rho=reshape(R*v,q.N,q.N)
        drho=reshape(R*(im.*ell.*v),q.N,q.N)
        b[i,j],rep=bures_surrogate(rho,drho)
        repair[i,j]=rep ? 1 : 0
    end
    save_wide(joinpath(folder,"bures_absolute.csv"),"theta_over_pi",theta./pi,btimes,b)
    save_wide(joinpath(folder,"bures_repaired_mask.csv"),"theta_over_pi",theta./pi,btimes,repair)
    if sum(repair)>0
        @warn "Bures diagnostic used eigenvalue clipping; not physical Bures metric" samples=Int(sum(repair))
    end
    # Independently fit microscopic spectrum for use in Fig2.
    pos=findall(j->ell[j]>=1 && ell[j]<=min(q.K,5),eachindex(ell))
    length(pos)>=2 || error("K>=2 required for diffusion fit")
    x=Float64.(ell[pos]).^2;y=-real.(Z[pos])
    D=2*dot(x,y)/dot(x,x)
    open(joinpath(folder,"metadata.txt"),"w") do io
        for (k,val) in ["N"=>q.N,"K"=>q.K,"Delta"=>q.Delta,
                        "epsilon"=>q.epsilon,"gamma_up"=>q.gamma_up,
                        "gamma_down"=>q.gamma_down,"nu"=>q.nu,
                        "omega_cycle"=>omega,"D_fit"=>D]
            println(io,"$k=$val")
        end
        println(io,"Bures is a surrogate reconstructed-matrix diagnostic, not physical QFI")
    end
    @printf("Fitted diffusion coefficient D_tilde = %.9g\n",D)
    return D
end
