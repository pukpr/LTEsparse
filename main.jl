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

# Check if file has a header by test-reading (handles whitespace/comma delimiting and strips whitespace)
test_df = CSV.read(data_path, DataFrame; limit=1, delim=' ', ignorerepeated=true, stripwhitespace=true)

times = Float64[]
data = Float64[]

if "year_fraction" in names(test_df) && "nino34_normalized" in names(test_df)
    local df = CSV.read(data_path, DataFrame; delim=' ', ignorerepeated=true, stripwhitespace=true)
    global times = Float64.(df.year_fraction)
    global data  = Float64.(df.nino34_normalized)
else
    # Fallback for comma if first check failed due to comma instead of space
    test_df_comma = CSV.read(data_path, DataFrame; limit=1, stripwhitespace=true)
    if "year_fraction" in names(test_df_comma) && "nino34_normalized" in names(test_df_comma)
        local df = CSV.read(data_path, DataFrame; stripwhitespace=true)
        global times = Float64.(df.year_fraction)
        global data  = Float64.(df.nino34_normalized)
    else
        # Assume 1st column is year, 2nd column is amplitude (headerless)
        # We'll use CSV with auto-detection but force ignorerepeated in case of arbitrary whitespace
        # We default to space, ignorerepeated, stripwhitespace
        try
            local df = CSV.read(data_path, DataFrame; header=false, delim=' ', ignorerepeated=true, stripwhitespace=true)
            if size(df, 2) < 2
                throw(ArgumentError("Less than 2 columns found with space delimiter."))
            end
            global times = Float64.(df[:, 1])
            global data  = Float64.(df[:, 2])
        catch
            # Fallback to comma separation
            local df = CSV.read(data_path, DataFrame; header=false, stripwhitespace=true)
            global times = Float64.(df[:, 1])
            global data  = Float64.(df[:, 2])
        end
    end
end

println("Using config: ", config_path, " and data: ", data_path)
# 3. Invoke the Discovery Engine
println("Starting LTE Discovery for ENSO...")
run_discovery(config_path, times, data)

println("Process Complete. Check versioned JSON for updated folding factors.")
