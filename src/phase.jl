# Convert principal-value angles into continuous angles along a trajectory.
function unwrap_angles(phi::AbstractVector{<:Real})
    result=Float64.(phi)
    for k in 2:length(result)
        difference=Float64(phi[k]-phi[k-1])
        difference=mod(difference+pi,2pi)-pi
        result[k]=result[k-1]+difference
    end
    return result
end

# The classical uniform phase is the continuous branch of
# atan2(sqrt(Delta-epsilon)*p,sqrt(Delta+epsilon)*x).
# This is equivalent to sqrt(Delta^2-epsilon^2)*int dphi/(Delta+epsilon*cos(2phi)).
function uniform_phase(xs,ps,Delta,epsilon)
    Delta>abs(epsilon) || error("This test assumes Delta > |epsilon| > = 0")
    raw=[atan(sqrt(Delta-epsilon)*ps[k],
              sqrt(Delta+epsilon)*xs[k]) for k in eachindex(xs)]
    theta=unwrap_angles(raw)
    return theta .- theta[1]
end

function cycle_samples(q,period,npoints)
    # Relax classical trajectory and synchronize with positive-x, p=0 crossing.
    dt=min(0.007,period/1200)
    rr=sqrt(max(1.0,2*(q.gamma_up-q.gamma_down)/q.nu))
    x,p=rr,0.0
    for _ in 1:ceil(Int,150/dt)
        x,p=rk4(x,p,dt,q)
    end
    crossings=0
    for _ in 1:ceil(Int,3*period/dt)
        xx,pp=x,p
        x,p=rk4(x,p,dt,q)
        if pp>0 && p<=0 && x>0
            # Linear interpolation onto the section p=0.
            frac=pp/(pp-p)
            x=xx+frac*(x-xx);p=0.0
            crossings+=1
            break
        end
    end
    crossings==1 || error("Could not locate the positive-x section")
    # Uniformly sample time across one complete orbit, including the endpoint.
    t=collect(range(0.0,period,length=npoints))
    xs=zeros(npoints);ps=zeros(npoints)
    xs[1]=x;ps[1]=p
    hmax=min(0.006,period/1400)
    for j in 2:npoints
        nsteps=max(1,ceil(Int,(t[j]-t[j-1])/hmax))
        hh=(t[j]-t[j-1])/nsteps
        for _ in 1:nsteps
            x,p=rk4(x,p,hh,q)
        end
        xs[j]=x;ps[j]=p
    end
    return t,xs,ps
end

function complex_coherent_density(N,x,p)
    alpha=(x+im*p)/sqrt(2)
    psi=coherent(N,alpha)
    n2=sum(abs2,psi)
    n2>1e-12 || error("Coherent state too far outside Fock truncation: alpha=$alpha")
    psi./=sqrt(n2)
    return vec(psi*psi')
end

function export_phase(q,basis,harmonics,folder,npoints)
    N=q.N; Ktest=maximum(harmonics)
    Delta=q.Delta; eps=q.epsilon; up=q.gamma_up; down=q.gamma_down; nu=q.nu
    Ktest>=1 || error("Ktest must be at least one")
    npoints>=12 || error("At least 12 phase samples are required")
    mkpath(folder)
    # Reuse exactly the same physical model and cached left eigenvectors.
    expected_n=(up-down)/(2nu)
    N < expected_n+5sqrt(max(expected_n,1.0)) &&
        @warn "Fock cutoff may be insufficient for typical limit-cycle occupation" N expected_n
    period,omega_numerical=estimate_period(q)
    omega_exact=sqrt(Delta^2-eps^2)
    @printf("Classical numerical omega %.9g; analytic omega %.9g\n",omega_numerical,omega_exact)
    times,xs,ps=cycle_samples(q,period,npoints)
    theta=uniform_phase(xs,ps,Delta,eps)
    @printf("Classical phase change over one period: %.8f pi\n",(theta[end]-theta[1])/pi)
    phase_drift=maximum(abs.(theta .+ omega_exact.*times))
    @printf("Classical exact-phase vs time consistency max error: %.4e rad\n",phase_drift)

    Ktest <= q.K || error("Phase harmonics exceed cached spectral cutoff")
    Z=[basis.Z[q.K+1+l] for l in 1:Ktest]
    U=[basis.U[:,l] for l in 1:Ktest]

    coeff=zeros(ComplexF64,npoints,Ktest)
    for j in 1:npoints
        state=complex_coherent_density(N,xs[j],ps[j])
        for l in harmonics
            coeff[j,l]=dot(U[l],state)
        end
    end
    raw_amplitudes=abs.(coeff)
    # The arbitrary complex phase of each left eigenvector cancels when
    # calculating arg(c_l(phi)/c_l(phi_0)).
    quantum_phase=zeros(npoints,Ktest)
    phase_errors=zeros(npoints,Ktest)
    amplitudes=zeros(npoints,Ktest)
    for l in harmonics
        raw=angle.(coeff[:,l])
        unwrapped=unwrap_angles(raw)
        quantum_phase[:,l].= (unwrapped .- unwrapped[1])./l
        # Fully unwrapped difference: does not conceal a winding-number mismatch.
        phase_errors[:,l].=quantum_phase[:,l].-theta
        amplitudes[:,l].=raw_amplitudes[:,l]
    end
    headers=["t","phi_unwrapped","theta_classical","theta_uniform_time",
             "x","p"]
    phi=unwrap_angles(atan.(ps,xs))
    table=hcat(times,phi,theta,-omega_exact.*times,xs,ps)
    for l in harmonics
        push!(headers,"theta_quantum_l$l")
        push!(headers,"phase_error_l$l")
        push!(headers,"amplitude_l$l")
        push!(headers,"amplitude_ratio_l$l")
        table=hcat(table,quantum_phase[:,l],phase_errors[:,l],amplitudes[:,l],
                   raw_amplitudes[:,l]./max(raw_amplitudes[1,l],1e-300))
    end
    write_csv(joinpath(folder,"theta_comparison.csv"),headers,table)
    ampheaders=["t","theta_classical"]
    ampdata=hcat(times,theta)
    for l in harmonics
        push!(ampheaders,"abs_c_l$l")
        ampdata=hcat(ampdata,amplitudes[:,l])
    end
    write_csv(joinpath(folder,"harmonic_amplitudes.csv"),ampheaders,ampdata)
    open(joinpath(folder,"summary.txt"),"w") do io
        println(io,"Phase and amplitude comparisons use coherent-state overlaps with the left modes.")
        @printf(io,"N=%d Ktest=%d Delta=%g epsilon=%g nu=%g\n",N,Ktest,Delta,eps,nu)
        @printf(io,"omega_classical_exact=%.12g\n",omega_exact)
        @printf(io,"omega_classical_numerical=%.12g\n",omega_numerical)
        @printf(io,"max_abs_classical_phase_time_error=%.8g\n",phase_drift)
        for l in harmonics
            mask=raw_amplitudes[:,l] .> maximum(raw_amplitudes[:,l])*1e-3
            rms=sqrt(sum(abs2,phase_errors[mask,l])/sum(mask))
            peak=maximum(abs.(phase_errors[mask,l]))
            cv=std(raw_amplitudes[:,l])/max(mean(raw_amplitudes[:,l]),1e-20)
            drift=quantum_phase[end,l]-quantum_phase[1,l]
            @printf(io,"l=%d total_phase_winding=%.8g*pi\n",l,drift/pi)
            @printf(io,"l=%d z=%.9g%+.9gi  quantum_phase_RMS_error=%.8g rad  max_error=%.8g  amplitude_CV=%.8g\n",l,real(Z[l]),imag(Z[l]),rms,peak,cv)
            @printf("l=%d: RMS phase error %.5g rad; amplitude CV %.5g\n",l,rms,cv)
        end
    end
    open(joinpath(folder,"comparison_parameters.txt"),"w") do io
        for (k,v) in ["N"=>N,"Ktest"=>Ktest,"Delta"=>Delta,
                      "epsilon"=>eps,"gamma_up"=>up,"gamma_down"=>down,
                      "nu"=>nu,"omega_exact"=>omega_exact,
                      "omega_numerical"=>omega_numerical]
            println(io,"$k=$v")
        end
    end
    println("Saved phase comparison: ",abspath(folder))
end

