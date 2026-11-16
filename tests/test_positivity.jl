using Test,LinearAlgebra,Printf,DelimitedFiles
include(joinpath(@__DIR__,"..","src","model.jl"))
include(joinpath(@__DIR__,"..","src","positivity.jl"))

function basis_for(rho,V;gamma=.02)
    ns=norm(rho)
    nv=norm(V)
    ell=[-1,0,1]
    R=hcat(vec(V'/nv),vec(rho/ns),vec(V/nv))
    c=ComplexF64[nv,ns,nv]
    Z=ComplexF64[-gamma+.8im,0,-gamma-.8im]
    (;rho,ell,R,c,Z)
end

@testset "Gaussian initialization in the normalized mode frame" begin
    stationary=ComplexF64[.5 0;0 .5]
    raw=basis_for(stationary,ComplexF64[0 .75;0 0])
    b=gaussian_basis(raw,2)
    o=PositivityOptions(sigma=2.0)
    result=restore_reference(raw,b,2,o,[1.,5.])
    threshold=log(norm(stationary)/.5)
    @test threshold-1e-8<=result.precision<=threshold+.01
    @test result.sigma<o.sigma
    expected=exp.(-raw.ell.^2 ./ (2*result.sigma^2))
    @test gaussian_weights(b,result.precision,0.)≈expected/sum(expected)
    @test b.R*b.R'≈raw.R*raw.R'

    # Independent rescalings and rephasings of the cached right eigenvectors.
    scale=ComplexF64[2im,-3,4cis(.7)]
    rescaled=merge(raw,(R=raw.R.*reshape(scale,1,:),c=raw.c./scale))
    bscaled=gaussian_basis(rescaled,2)
    @test bscaled.R*bscaled.R'≈b.R*b.R'
    t,theta=1.2,.37
    rho,drho=gaussian_pair(b,2,result.precision,t,theta)
    @test first(gaussian_pair(bscaled,2,result.precision,t,theta))≈rho
    h=1e-5
    rp=first(gaussian_pair(b,2,result.precision,t,theta+h))
    rm=first(gaussian_pair(b,2,result.precision,t,theta-h))
    @test drho≈(rp-rm)/(2h) atol=1e-9

    for t in result.times, theta in range(0,2pi,length=513)
        rho,_=gaussian_pair(b,2,result.precision,t,theta)
        @test abs(tr(rho)-1)<1e-12
        @test norm(rho-rho')<1e-12
        @test eigmin(Hermitian(rho))>=-o.psd_tolerance
    end
    narrow=PositivityOptions(sigma=.4)
    @test restore_reference(raw,b,2,narrow,[0.]).sigma≈.4

    # The default width matches the raw second moment before checking positivity.
    small=basis_for(stationary,ComplexF64[0 .2;0 0])
    bs=gaussian_basis(small,2)
    beta=match_precision(bs,fs_metric(projected_weights(small,0.),small.ell))
    @test dot(gaussian_weights(bs,beta,0.),bs.ell.^2)≈
        dot(abs2.(small.c),small.ell.^2)/sum(abs2,small.c)
    @test restore_reference(small,bs,2,PositivityOptions(),[0.]).precision≈beta

    # A non-Gaussian reference with two differently damped harmonic pairs.
    ell=collect(-2:2)
    R=zeros(ComplexF64,9,5)
    for (col,row) in [(1,3),(2,2),(4,4),(5,7)]
        R[row,col]=1
    end
    R[:,3]=vec(Matrix{ComplexF64}(I,3,3)/sqrt(3))
    raw5=(;ell,R,c=ComplexF64[.10,.12,1/sqrt(3),.12,.10],Z=-.06 .* ell.^2 .- .8im .* ell)
    tau=3.0
    b5=gaussian_basis(raw5,3;phase_time=tau)
    w=abs2.(raw5.c).*exp.(2tau.*real.(raw5.Z))
    target=dot(w,ell.^2)/sum(w)
    beta_tau=match_precision(b5,target)
    @test dot(gaussian_weights(b5,beta_tau,0.),ell.^2)≈target atol=1e-12
    @test abs(dot(gaussian_weights(b5,beta_tau,tau),ell.^2)-target)>1e-3
    b5_seed=gaussian_basis(raw5,3)
    @test b5.c≈b5_seed.c.*cis.(tau.*imag.(raw5.Z))
    @test gaussian_match_time(narrow,[1.,5.])==1.
    timed=restore_reference(raw5,b5,3,PositivityOptions(match_time=tau),[1.,5.])
    @test timed.match_time==tau
    @test timed.times==[0.,1.,5.]
    @test timed.trial_variance≈target atol=1e-12
    @test timed.selected_variance≈target atol=1e-12

    # A negative minimum between coarse angular samples must fail certification.
    shifted=basis_for(stationary,cis(-pi/8)*ComplexF64[.251 0;0 -.251])
    bg=gaussian_basis(shifted,2)
    beta=log(norm(stationary)/norm(ComplexF64[.251 0;0 -.251]))
    coarse=PositivityOptions(theta_grid=8,theta_grid_max=8)
    sampled=[eigmin(Hermitian(first(gaussian_pair(bg,2,beta,0.,2pi*k/8)))) for k in 0:7]
    @test minimum(sampled)>0
    @test eigmin(Hermitian(first(gaussian_pair(bg,2,beta,0.,pi/8))))<0
    @test !certify_gaussian(bg,prepare_certificate(bg,2,coarse),coarse,beta,[0.]).ok
end
