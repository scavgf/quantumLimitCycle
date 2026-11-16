# Maintained geometry exports. model.jl's legacy geometry exporter is not used.
function physical_bures(rho,drho; o, denominator_tolerance=1e-12)
    norm(rho-rho')<=o.trace_tolerance || error("Non-Hermitian Bures input")
    norm(drho-drho')<=o.trace_tolerance || error("Non-Hermitian phase derivative")
    abs(tr(rho)-1)<=o.trace_tolerance || error("Bures input is not trace one")
    E=eigen(Hermitian((rho+rho')/2)); values=E.values
    minimum(values)>=-o.psd_tolerance || error("Nonpositive Gaussian Bures input. Try a narrower sigma or improve solver accuracy.")
    A=E.vectors'*drho*E.vectors
    g=0.
    # No clipping or density replacement. Near-null support is handled explicitly.
    for j in eachindex(values),i in eachindex(values)
        den=values[i]+values[j]
        if den>denominator_tolerance
            g+=abs2(A[i,j])/(2den)
        elseif abs(A[i,j])>max(10o.psd_tolerance,1e-10)
            error("Phase derivative has a resolved component in numerically null Bures support. Improve cutoff/precision; no surrogate curve was substituted.")
        end
    end
    g,minimum(values)
end
function export_positive_geometry(q,basis,omega,folder,o,restoration; bures_time=4.0)
    ell=basis.ell; Z=basis.Z
    precision=restoration.precision
    times=sort(unique(q.times)); T=length(times)
    length(times)>0 || error("Specify at least one observation time")
    W=hcat([gaussian_weights(basis,precision,t) for t in times]...)
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
    metric=[fs_metric(gaussian_weights(basis,precision,t),ell) for t in tg]
    write_csv(joinpath(folder,"metric_time.csv"),["t","g_FS"],hcat(tg,metric))
    # Only Fig. 1 needs this costly angular reconstruction.
    # Its time is independent of the Fig. 2 selected times.
    btimes=[bures_time]
    theta=collect(0:q.nangle-1).*2pi/q.nangle
    b=zeros(q.nangle,length(btimes));repair=zeros(q.nangle,length(btimes))
    minlam=similar(b);traceerror=similar(b);hermiticity=similar(b)
    for (j,t) in enumerate(btimes), (i,th) in enumerate(theta)
        rho,drho=gaussian_pair(basis,q.N,precision,t,th)
        b[i,j],minlam[i,j]=physical_bures(rho,drho;o)
        traceerror[i,j]=abs(tr(rho)-1)
        hermiticity[i,j]=norm(rho-rho')
        repair[i,j]=0.
    end
    save_wide(joinpath(folder,"bures_absolute.csv"),"theta_over_pi",theta./pi,btimes,b)
    save_wide(joinpath(folder,"bures_repaired_mask.csv"),"theta_over_pi",theta./pi,btimes,repair)
    save_wide(joinpath(folder,"bures_min_eigenvalue.csv"),"theta_over_pi",theta./pi,btimes,minlam)
    save_wide(joinpath(folder,"bures_trace_error.csv"),"theta_over_pi",theta./pi,btimes,traceerror)
    save_wide(joinpath(folder,"bures_hermiticity_error.csv"),"theta_over_pi",theta./pi,btimes,hermiticity)

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
        println(io,"time_origin=Gaussian start")
        println(io,"gaussian_start_projection_time=$(restoration.match_time)")
        println(io,"gaussian_sigma=$(restoration.sigma)")
        println(io,"positivity_scope=$(o.scope)")
        println(io,"psd_tolerance=$(o.psd_tolerance)")
        println(io,"Bures is calculated from the Gaussian-initialized density family and its own phase derivative; no clipping.")
    end
    @printf("Fitted diffusion coefficient D_tilde = %.9g\n",D)
    return D
end
