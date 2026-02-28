using Zygote, LinearAlgebra, Statistics, JSON3, Dates, Plots

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
   
    # Load bias if it exists, otherwise default to 0.0
    bias = haskey(raw, :bias) ? Float64(raw.bias) : 0.0
    p = (c=c_init, β=Float64.(raw.beta_weights), A=Float64.(raw.wavenumber_gains), θ=Float64.(raw.mod_phases), bias=bias)
    return p, ω_basis, metadata, raw
end

# --- 2. Differentiable Forward Pass ---
function lte_forward(p, t, ω_basis)
    # The 'Wiggly' Latent Signal
    z = sum(real(p.c[j] * exp(im * 2π * ω_basis[j] * t)) for j in 1:length(ω_basis))
    # The Exact Trig Folding (Severe Modulation) + Explicit Bias Offset
    return sum(p.β[k] * sin(p.A[k] * z + p.θ[k]) for k in 1:length(p.β)) + p.bias
end

# --- 3. The Discovery Loop (Backprop + SINDy) ---
function run_discovery(config_path, times, data)
    p, ω_basis, metadata, original = load_and_flatten(config_path)
    train_idx = 1:floor(Int, 0.8 * length(times))
   
    best_p = p
    min_val_mse = Inf
    best_correlation_val = 0.0
    best_correlation_train = 0.0
    best_preds_full = zeros(length(times))

    for epoch in 1:5000
        # Zygote AD Heavy Lifting
        val, grads = withgradient(p_opt -> begin
            preds = [lte_forward(p_opt, t, ω_basis) for t in times[train_idx]]
            mse = mean((preds .- data[train_idx]).^2)
            # Regularization: Sparsity on Tides (L1), Smoothness on Folding (L2)
            return mse + 0.02*sum(abs.(p_opt.c)) + 0.005*sum(p_opt.A.^2)
        end, p)

        # Gradient Clipping to prevent explosion
        clip_max = 5.0
        clip_norm(g) = begin
            n = norm(g)
            if isnan(n) || isinf(n)
                return zero.(g)
            else
                return n > clip_max ? g .* (clip_max / n) : g
            end
        end
        gc = clip_norm(grads[1].c)
        gβ = clip_norm(grads[1].β)
        gA = clip_norm(grads[1].A)
        gθ = clip_norm(grads[1].θ)
        gbias = grads[1].bias # Do not clip bias gradient, it needs freedom to move

        # Update Step (Adaptive Gradient Scaling)
        η = 0.01 / (norm(gc) + 1e-6)
        p = (c = p.c .- η .* gc,
             β = p.β .- η .* gβ,
             A = p.A .- (η*0.1) .* gA, # Slower updates for severe A
             θ = p.θ .- η .* gθ,
             bias = p.bias - (η*100.0) * gbias) # Bias needs a much faster learning rate to anchor properly

        # SINDy-style Pruning: Threshold based on relative power
        threshold = 0.005 * maximum(abs.(p.c))
        p.c[abs.(p.c) .< threshold] .= 0.0 + 0.0im

        # Cross-Validation
        v_preds = [lte_forward(p, t, ω_basis) for t in times[last(train_idx)+1:end]]
        v_mse = mean((v_preds .- data[last(train_idx)+1:end]).^2)
        correlation_val = std(v_preds) > 0 ? cor(v_preds, data[last(train_idx)+1:end]) : 0.0
        # Calculate the Pearson correlation for the validation set (the final objective measure)
        println("$(epoch) Correlation (R): $(round(correlation_val, digits=4))")

        if v_mse < min_val_mse
            min_val_mse = v_mse
            best_p = p
            best_correlation_val = correlation_val
            
            # Also capture the training correlation for the best model
            t_preds = [lte_forward(p, t, ω_basis) for t in times[train_idx]]
            best_correlation_train = std(t_preds) > 0 ? cor(t_preds, data[train_idx]) : 0.0
            
            # Capture full predictions for plotting
            best_preds_full = [lte_forward(p, t, ω_basis) for t in times]
        elseif epoch > 1500 && (v_mse > min_val_mse * 1.2)  # 1.05
            break # Early stopping on overfitting
        end
    end

    println("Discovery Complete. Best Validation MSE: $(round(min_val_mse, digits=6)) | Corresponding Validation Correlation (R): $(round(best_correlation_val, digits=4))")
    
    # Generate the time-series plot
    cv_start_time = times[last(train_idx)+1]
    cv_end_time = times[end]
    
    p_plot = plot(times, data, label="Target Data", color=:black, linewidth=1.5, title="LTE Discovery: Target vs Prediction\nTrain R: $(round(best_correlation_train, digits=3)) | CV R: $(round(best_correlation_val, digits=3))", xlabel="Time", ylabel="Value", legend=:outertopright)
    plot!(p_plot, times, best_preds_full, label="Best Prediction", color=:blue, linewidth=1.5)
    
    # Shade the CV region
    vspan!(p_plot, [cv_start_time, cv_end_time], color=:gray, alpha=0.3, label="CV Interval")
    
    savefig(p_plot, replace(config_path, ".json" => "_discovery_plot.png"))

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
        "beta_weights" => p.β, "wavenumber_gains" => p.A, "mod_phases" => p.θ, "bias" => p.bias,
        "metrics" => Dict("val_mse" => mse, "timestamp" => ts)
    )
   
    # Versioning
    cp(path, replace(path, ".json" => "_v$(ts).json"), force=true)
    open(path, "w") do io ; JSON3.pretty(io, final) ; end
end