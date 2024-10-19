using Random
using Interpolations
using GLMakie
using Statistics
using ProgressMeter

function gillespie(n0::Int64, tf::Float64, birth_rate::Float64, death_rate::Float64, inmigration_rate::Float64)
    n::Vector{Integer} = [n0]
    t::Vector{Float64} = [0.0]
    t1::Float64 = 0.0
    while t1 < tf
        if n[end] == 0
            append!(n, 0)
            append!(t, tf)
            break
        end
        r1::Float64 = rand()
        r2::Float64 = rand()
        a::Float64 = birth_rate*n[end]
        b::Float64 = death_rate*n[end]
        c::Float64 = inmigration_rate
        tau::Float64 = -log(r1)/(a+b+c) # sampleo una exponencial de parámetro a+b
        t1 += tau
        if r2 < (a+c)/(a+b+c)
            append!(n, n[end]+1)
        else
            append!(n, n[end]-1)
        end
        append!(t, t1)
    end
    return n, t
end

# function gillespie_con_transmutacion(n0, m0, split_rate, n_death_rate, m_transmute_rate, tf)
#     n = [n0]
#     m = [m0]
#     t = [0.0]
#     t1 = 0.0
#     whil

function plot_gillespie(birth_rate, death_rate, n_realizations, final_t, n₀; inmigration_rate=0.0, fig=Figure(), axis_index=(1, 1))
    standard_t = 0:0.1:final_t
    interp_vals = zeros(n_realizations, length(standard_t))
    @inbounds @showprogress for i in 1:n_realizations
        n, t = gillespie(n₀, final_t, birth_rate, death_rate, inmigration_rate)
        interp = interpolate((t, ), n, Gridded(Constant{Previous}()))
        interp_vals[i, :] = interp(standard_t)
    end
    title = "Tasa de nacimiento: $birth_rate, Tasa de muerte: $death_rate"
    i, j = axis_index
    ax = Axis(fig[i, j], xlabel = "Tiempo", ylabel = "Tamaño de Población", title=title)
    mean_vals = mean(interp_vals, dims=1)[1:end]
    std_vals = (std(interp_vals, dims=1))[1:end]
    lines!(ax, standard_t, mean_vals, color=:orange, label="Estimación promedio") 
    sol_analitica = n₀.*exp.((birth_rate-death_rate).*standard_t) .- inmigration_rate/(birth_rate - death_rate) .+ inmigration_rate/(birth_rate - death_rate) .* exp.((birth_rate-death_rate).*standard_t)
    lines!(ax, standard_t, sol_analitica, color=:blue, label="Solución analítica")

    if inmigration_rate == 0.0
        var_analitica = n₀.*((birth_rate + death_rate)/(birth_rate - death_rate)).*(exp.(2 .*(birth_rate-death_rate).*standard_t) - exp.((birth_rate-death_rate).*standard_t))
        lines!(ax, standard_t, sol_analitica + var_analitica, color=:blue, alpha=0.3, label="Varianza analítica")
        lines!(ax, standard_t, sol_analitica - var_analitica, color=:blue, alpha=0.3)
    end
    band!(ax, standard_t, mean_vals + std_vals.^2, mean_vals - std_vals.^2, color=:orange, alpha=0.3, label="Varianza Muestral")
    axislegend(ax, position=:rb)
    return fig
end

function evaluar_procesos()
    final_t = 100.0
    n₀ = 10
    n_realizations = 1_000
    birth_rates = [0.45, 0.50, 0.55]
    death_rates = [0.5]
    inmigration_rate = 1.5
    fig = Figure()
    for (i, b) in enumerate(birth_rates)
        for (j, d) in enumerate(death_rates)
            plot_gillespie(b, d, n_realizations, final_t, n₀; inmigration_rate=inmigration_rate, fig=fig, axis_index=(i, j))
        end
    end
    display(fig)
end

