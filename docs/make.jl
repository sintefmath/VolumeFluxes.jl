# Copyright (c) 2024 SINTEF AS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

using Documenter, Literate, VolumeFluxes, Parameters

examples_dir = realpath(joinpath(@__DIR__, "..", "examples"))


function process_includes(content)
    # Replace include statements with actual content
    while (m = match(r"include\(\"([^\"]+)\"\)", content)) !== nothing
        include_path = joinpath(examples_dir, m.captures[1])
        include_content = read(include_path, String)
        content = replace(content, m.match => include_content)
    end    
    return content
end
#Literate.markdown("examples/urban.jl", "docs/src/"; execute=false, preprocess=process_includes)
for example in ("terrain", "optimization", "shallow_water_1d", "callbacks")
    Literate.markdown(joinpath(examples_dir, "$example.jl"), joinpath(@__DIR__, "src");
        execute=false, flavor=Literate.CommonMarkFlavor(), preprocess=process_includes)
end

makedocs(modules = [VolumeFluxes], 
    root=@__DIR__,
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://sintefmath.github.io/VolumeFluxes.jl/"),
    sitename="VolumeFluxes.jl",
    draft=false,
    pages = [
        "Introduction" => "index.md",
        "Examples" => ["shallow_water_1d.md",
                       "terrain.md",
                       "optimization.md",
                       "callbacks.md"],
        "Index" => "indexlist.md",
        "Public API" => "api.md"
    ])
touch(joinpath(@__DIR__, "build", ".nojekyll"))
