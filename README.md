# VolumeFluxes.jl
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