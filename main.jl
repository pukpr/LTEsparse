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

# Check if file has a header by test-reading
test_df = CSV.read(data_path, DataFrame; limit=1)
if "year_fraction" in names(test_df) && "nino34_normalized" in names(test_df)
    df = CSV.read(data_path, DataFrame)
    times = Float64.(df.year_fraction)
    data  = Float64.(df.nino34_normalized)
else
    # Assume 1st column is year, 2nd column is amplitude (headerless)
    df = CSV.read(data_path, DataFrame; header=false)
    times = Float64.(df[:, 1])
    data  = Float64.(df[:, 2])
end

println("Using config: ", config_path, " and data: ", data_path)
# 3. Invoke the Discovery Engine
println("Starting LTE Discovery for ENSO...")
run_discovery(config_path, times, data)

println("Process Complete. Check versioned JSON for updated folding factors.")
