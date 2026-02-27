using CSV, DataFrames

# Load your discovery engine logic
include("ls.jl")

# 1. Load the Target Time Series (e.g., ENSO Index)
df = CSV.read("mf.dat", DataFrame)
times = Float64.(df.year_fraction)
data  = Float64.(df.nino34_normalized)

# 2. Path to your initial JSON Parameters
config_path = "simple.json"

# 3. Invoke the Discovery Engine
println("Starting LTE Discovery for ENSO...")
run_discovery(config_path, times, data)

println("Process Complete. Check versioned JSON for updated folding factors.")
