using Documenter
using ExpExp

DocMeta.setdocmeta!(ExpExp, :DocTestSetup, :(using ExpExp); recursive=true)

makedocs(;
    modules=[ExpExp],
    authors="Guo Chu",
    sitename="ExpExp.jl",
    doctest=true,
    clean=true,
    checkdocs=:exports,
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https://guochu.github.io/ExpExp/",
        assets=String[],
    ),
    pages=[
        "首页" => "index.md",
        "指南" => [
            "算法" => "man/algorithms.md",
            "示例" => "man/examples.md",
            "精度与效率" => "man/performance.md",
        ],
        "API 参考" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/guochu/ExpExp",
    devbranch="master",
    push_preview=true,
)
