using ExpExp
using LinearAlgebra
using Test

@testset "ExpExp.jl" begin
	@testset "exponential_expansion" begin
		atol = 1.0e-10
		datas = [
			[1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125],
			[0.5^k + 2*0.7^k for k in 1:20],
			[-2*0.3^k + 0.9^k - 1.5*0.5^k for k in 1:20],
		]
		for ydata in datas
			for alg in (PronyExpansion(n=20, tol=atol), DeterminedPronyExpansion(n=20, tol=atol))
				xs, lambdas = exponential_expansion(ydata, alg)
				@test expansion_error(ydata, xs, lambdas) / norm(ydata) < atol
			end
		end
	end

	@testset "exponential_expansion: function input" begin
		ydata = [0.5^k + 2*0.7^k for k in 1:20]
		atol = 1.0e-8
		xs, lambdas = exponential_expansion(k -> 0.5^k + 2*0.7^k, 20, PronyExpansion(n=20, tol=atol))
		@test expansion_error(ydata, xs, lambdas) / norm(ydata) < atol
		xs, lambdas = exponential_expansion(k -> 0.5^k + 2*0.7^k, 20, alg=PronyExpansion(n=20, tol=atol))
		@test expansion_error(ydata, xs, lambdas) / norm(ydata) < atol
	end

	@testset "exponential_expansion: stepsize" begin
		ydata = [0.5^k + 2*0.7^k for k in 1:20]
		stepsize = 3
		atol = 1.0e-8
		xs, lambdas = exponential_expansion(ydata[1:stepsize:end], PronyExpansion(n=20, stepsize=stepsize, tol=atol))
		@test expansion_error(ydata, xs, lambdas) / norm(ydata) < atol
	end

	@testset "expansion_error: concatenated form" begin
		ydata = [0.5^k for k in 1:10]
		err = expansion_error(ydata, vcat([1.0], [0.5]))
		@test err ≈ 0 atol=1.0e-12
	end

	@testset "MatrixPencilExpansion" begin
		using Random
		atol = 1.0e-10
		datas = [
			[0.5^k + 2*0.7^k for k in 1:20],
			[-2*0.3^k + 0.9^k - 1.5*0.5^k for k in 1:20],
		]
		for ydata in datas
			xs, lambdas = exponential_expansion(ydata, MatrixPencilExpansion(n=20, tol=atol, verbosity=0))
			@test expansion_error(ydata, xs, lambdas) / norm(ydata) < atol
		end
	end

	@testset "MatrixPencil vs Prony cross-validation" begin
		ydata = [0.5^k + 2 * 0.7^k for k in 1:20]
		atol = 1.0e-10

		# both algorithms recover the same bases on exact exponential data
		xsm, lamm = exponential_expansion(ydata, MatrixPencilExpansion(n=20, tol=atol, verbosity=0))
		xsp, lamp = exponential_expansion(ydata, PronyExpansion(n=20, tol=atol, verbosity=0))
		@test sort(real(lamm)) ≈ sort(real(lamp)) rtol = 1.0e-6
		@test isapprox(sort(real(lamm)), [0.5, 0.7]; rtol=1e-6)

		# noise robustness: matrix pencil stays accurate, Prony does not
		Random.seed!(7)
		ynoisy = ydata .+ 1.0e-4 .* randn(20)
		xn, lamn = matrix_pencil(ynoisy, 2)
		@test expansion_error(ynoisy, xn, lamn) / norm(ynoisy) < 1.0e-2          # fits the noisy data well
		@test maximum(abs.(sort(real(lamn)) .- [0.5, 0.7])) < 1.0e-2             # bases stay stable
	end
end
