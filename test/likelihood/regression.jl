@testset "regresion" begin
    y = randn(10)
    l = GaussianLikelihood()
    @test AGP.treat_labels!(y, l) == (y, 1, l)
    @test predict_y(l, y) == y
    @test predict_y(l, [y]) == y
end
