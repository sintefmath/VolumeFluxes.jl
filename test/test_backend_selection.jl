@testset "Backend selection" begin
    @test !VolumeFluxes.cuda_enabled("cpu", () -> error("CPU mode must not probe CUDA"))
    for available in (false, true)
        @test VolumeFluxes.cuda_enabled("auto", () -> available) == available
    end
    @test VolumeFluxes.cuda_enabled("gpu", () -> true)
    @test_throws ErrorException VolumeFluxes.cuda_enabled("gpu", () -> false)
    @test_throws ErrorException VolumeFluxes.cuda_enabled("auto", () -> error("CUDA initialization failed"))
    for mode in ("", "GPU", "invalid")
        @test_throws ArgumentError VolumeFluxes.cuda_enabled(mode, () -> error("Must validate first"))
    end
    withenv("VOLUMEFLUXES_BACKEND" => "cpu") do
        @test !VolumeFluxes.has_cuda_backend()
        backends = get_available_backends()
        @test length(backends) == 1
        @test only(backends) isa VolumeFluxes.CPUBackend
    end
    withenv("VOLUMEFLUXES_BACKEND" => "invalid") do
        @test_throws ArgumentError VolumeFluxes.has_cuda_backend()
        @test_throws ArgumentError get_available_backends()
    end
end
