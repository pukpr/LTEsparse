using CSV, DataFrames

# Load your discovery engine logic
include("ls.jl")

# Parse Command Line Arguments
config_path = length(ARGS) >= 1 ? ARGS[1] : "simple.json"
data_path   = length(ARGS) >= 2 ? ARGS[2] : "mf.dat"

# 1. Load the Target Time Series (e.g., ENSO Index)
if !isfile(config_path) || !isfile(data_path)
    println("Usage: julia main.jl [config.json] [data.dat]")
    exit(1)
end

df = CSV.read(data_path, DataFrame)
times = Float64.(df.year_fraction)
data  = Float64.(df.nino34_normalized)

println("Using config: ", config_path, " and data: ", data_path)
# 3. Invoke the Discovery Engine
println("Starting LTE Discovery for ENSO...")
run_discovery(config_path, times, data)

println("Process Complete. Check versioned JSON for updated folding factors.")
