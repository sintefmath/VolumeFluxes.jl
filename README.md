# VolumeFluxes.jl

![VolumeFluxes logo ](docs/src/assets/volume_fluxes_small.png)

Finite volume solvers with an emphasis on modelling surface water with the shallow water equations.

## Setting up the first time

Setting up the project directly should just be (run from the root of the repository)

```bash
julia --project
] instantiate
```

It is probably a good idea to run all tests first verify everything installed correctly, so from the root of the repository, run

```bash
julia --project test/runtests.jl
```

### Test validation and GPU coverage

CI targets Julia 1.10 (a conservative baseline) and the latest stable Julia 1.x
on Linux; this is a validation matrix, not a claim of compatibility with all
intervening releases. The project previously declared no Julia compatibility range.
Pull requests, pushes to `main`, and manual **CPU tests** runs execute the full
test suite with CPU backends only. GPU portions of mixed test files are skipped,
not reported as GPU validation. CairoMakie uses its headless SVG backend and
diagnostic plot display is disabled by default.

From the repository root, after instantiation:

```bash
# Auto (default): CPU plus CUDA when CUDA.functional() is true
julia --project -e 'using Pkg; Pkg.test()'
# CPU only, without probing a CUDA driver/device
VOLUMEFLUXES_BACKEND=cpu julia --project -e 'using Pkg; Pkg.test()'
# CPU plus required CUDA; errors rather than silently falling back
VOLUMEFLUXES_BACKEND=gpu julia --project -e 'using Pkg; Pkg.test()'
```

`VOLUMEFLUXES_BACKEND` accepts only `auto`, `cpu`, or `gpu` and controls backend
availability checks. Explicit `make_cuda_backend()` calls still request CUDA.
The suite logs whether GPU coverage is included or skipped.

**GPU tests (manual)** is independent of CPU CI and runs only when a maintainer
dispatches it on `main`. No GPU runner is assumed or paid runner enabled.
To use it, register a Linux x64 self-hosted runner with the labels `self-hosted`,
`linux`, `x64`, and `cuda`, a CUDA.jl-supported NVIDIA GPU/driver, and network
access for Julia packages/artifacts. It must support the Node.js runtime used by
the actions; CUDA.jl normally downloads its own toolkit. Confirm
`CUDA.functional()` is true on that runner. Without a matching runner the manual
job queues; with nonfunctional CUDA it fails.

Use an isolated, preferably ephemeral runner with no sensitive credentials and
restrict runner access to trusted workflows. Never dispatch unreviewed code on
it. The workflow deliberately accepts no PR/ref input, checks out only `main`,
and never runs on `pull_request` or `pull_request_target`.

## Examples
We currently have a couple ofexamples for a full simulation scenario, see

  * `examples/shallow_water_1d.jl`
  * `examples/urban.jl`
  * `examples/terrain.jl`

## Generating the documentation

Run (note: currently takes a couple of minutes)

    julia --project docs/make.jl

To view the generated documentation, it is probably a good idea to install the `LiveServer` package and view said documentation through that

    julia -e 'using LiveServer; serve(dir="docs/build")'

then point your browser to `localhost:8000`.

The **Documentation** workflow validates builds on PRs with read-only permissions,
and publishes on pushes to `main` or manual runs on `main`. Literate examples are
converted without execution (no GPU simulations). The Pages artifact contains
`docs/build/index.html` and `.nojekyll`; canonical links target
<https://sintefmath.github.io/VolumeFluxes.jl/>. Local builds use file-style HTML
links; CI builds use directory-style URLs for Pages.

One-time maintainer setup: select **Settings → Pages → Build and deployment →
Source: GitHub Actions**, and allow deployment from `main` in the `github-pages`
environment. These repository settings are not changed by this implementation.
Only the separate, default-branch deployment job receives Pages/OIDC write
permissions; PRs and forks cannot publish.

## License information
MIT License

Copyright (c) 2024 SINTEF AS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.