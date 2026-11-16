using Test,LinearAlgebra,Printf
include(joinpath(@__DIR__,"..","src","positivity.jl"))
include(joinpath(@__DIR__,"..","src","positive_geometry.jl"))
@testset "Bures metric uses the filtered physical family" begin
    o=PositivityOptions()
    for theta in range(0,2pi,length=25)
        v=.2*cis(theta)
        rho=ComplexF64[.5 v;conj(v) .5]
        drho=ComplexF64[0 im*v;-im*conj(v) 0]
        g,m=physical_bures(rho,drho;o)
        @test g≈.04 atol=1e-12
        @test m≈.3 atol=1e-12
    end
    @test_throws ErrorException physical_bures(ComplexF64[1.1 0;0 -.1],zeros(ComplexF64,2,2);o)
    @test_throws ErrorException physical_bures(ComplexF64[1 0;0 0],ComplexF64[0 0;0 1];o)
    g,m=physical_bures(ComplexF64[1 0;0 0],zeros(ComplexF64,2,2);o)
    @test g==0
end
