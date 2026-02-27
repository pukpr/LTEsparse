using Zygote, LinearAlgebra, Statistics, JSON3, Dates

# --- Helper: Stroboscopic Metadata Tracking ---
struct SatMeta
    parent::String
    n::Int
    f_alias::Float64
end

# --- 1. Load and Flatten ---
function load_and_flatten(path)
    raw = JSON3.read(read(path, String))
    ω_basis = Float64[]
    c_init = ComplexF64[]
    metadata = SatMeta[]

    for tide in raw.tidal_constituents
        # Re-expand satellites if not already explicit, or load existing
        for sat in tide.satellites
            push!(ω_basis, Float64(sat.freq_alias))
            push!(c_init, sat.amp * exp(im * sat.phase))
            push!(metadata, SatMeta(tide.name, sat.n, sat.freq_alias))
        end
    end
   
    p = (c=c_init, β=Float64.(raw.beta_weights), A=Float64.(raw.wavenumber_gains), θ=Float64.(raw.mod_phases))
    return p, ω_basis, metadata, raw
end

# --- 2. Differentiable Forward Pass ---
function lte_forward(p, t, ω_basis)
    # The 'Wiggly' Latent Signal
    z = sum(real(p.c[j] * exp(im * 2π * ω_basis[j] * t)) for j in 1:length(ω_basis))
    # The Exact Trig Folding (Severe Modulation)
    return sum(p.β[k] * sin(p.A[k] * z + p.θ[k]) for k in 1:length(p.β))
end

# --- 3. The Discovery Loop (Backprop + SINDy) ---
function run_discovery(config_path, times, data)
    p, ω_basis, metadata, original = load_and_flatten(config_path)
    train_idx = 1:floor(Int, 0.8 * length(times))
   
    best_p = p
    min_val_mse = Inf
    # correlation = 0.0

    for epoch in 1:150
        # Zygote AD Heavy Lifting
        val, grads = withgradient(p_opt -> begin
            preds = [lte_forward(p_opt, t, ω_basis) for t in times[train_idx]]
            mse = mean((preds .- data[train_idx]).^2)
            # Regularization: Sparsity on Tides (L1), Smoothness on Folding (L2)
            return mse + 0.02*sum(abs.(p_opt.c)) + 0.005*sum(p_opt.A.^2)
        end, p)

        # Update Step (Adaptive Gradient Scaling)
        η = 0.01 / (norm(grads[1].c) + 1e-6)
        p = (c = p.c .- η .* grads[1].c,
             β = p.β .- η .* grads[1].β,
             A = p.A .- (η*0.1) .* grads[1].A, # Slower updates for severe A
             θ = p.θ .- η .* grads[1].θ)

        # SINDy-style Pruning: Threshold based on relative power
        threshold = 0.005 * maximum(abs.(p.c))
        p.c[abs.(p.c) .< threshold] .= 0.0 + 0.0im

        # Cross-Validation
        v_preds = [lte_forward(p, t, ω_basis) for t in times[last(train_idx)+1:end]]
        v_mse = mean((v_preds .- data[last(train_idx)+1:end]).^2)
        correlation = cor(v_preds, data[last(train_idx)+1:end])
        # Calculate the Pearson correlation for the validation set (the final objective measure)
        println("Correlation (R): $(round(correlation, digits=4))")

        if v_mse < min_val_mse
            min_val_mse = v_mse
            best_p = p
        elseif epoch > 50 && (v_mse > min_val_mse * 1.05)
            break # Early stopping on overfitting
        end
    end

    save_roundtrip(config_path, best_p, metadata, min_val_mse)
end

# --- 4. Re-Nesting and Versioned Save ---
function save_roundtrip(path, p, metadata, mse)
    ts = Dates.format(now(), "yyyy-mm-dd_HHMM")
    # Group by parent name
    constituents = Dict()
    for (i, m) in enumerate(metadata)
        if abs(p.c[i]) > 0
            if !haskey(constituents, m.parent) constituents[m.parent] = [] end
            push!(constituents[m.parent], Dict(
                "n" => m.n, "amp" => abs(p.c[i]),
                "phase" => angle(p.c[i]), "freq_alias" => m.f_alias
            ))
        end
    end

    final = Dict(
        "tidal_constituents" => [Dict("name"=>k, "satellites"=>v) for (k,v) in constituents],
        "beta_weights" => p.β, "wavenumber_gains" => p.A, "mod_phases" => p.θ,
        "metrics" => Dict("val_mse" => mse, "timestamp" => ts)
    )
   
    # Versioning
    cp(path, replace(path, ".json" => "_v$(ts).json"), force=true)
    open(path, "w") do io ; JSON3.pretty(io, final) ; end
end