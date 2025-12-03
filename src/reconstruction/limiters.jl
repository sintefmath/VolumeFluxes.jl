
struct MinmodLimiter <: Limiter
    theta::Float64
end

struct VanLeerLimiter <: Limiter
end

struct MCLimiter <: Limiter
end

struct SuperbeeLimiter <: Limiter
end


#####################
# Basic building blocks
#####################

# 3-argument minmod
function minmod(a, b, c)
    if (a > 0) && (b > 0) && (c > 0)
        return min(a, b, c)
    elseif (a < 0) && (b < 0) && (c < 0)
        return max(a, b, c)
    end
    return zero(a)
end

# 2-argument minmod
function minmod2(a, b)
    if (a > 0) && (b > 0)
        return min(a, b)
    elseif (a < 0) && (b < 0)
        return max(a, b)
    end
    return zero(a)
end

function maxmod2(a, b)
    if a * b > 0
        return max(abs(a), abs(b))
    end
    return zero(a)
end

# van Leer limiter: Lim(a,b) = (a|b| + |a|b) / (|a| + |b|)
function vanleer(a, b)
    denom = abs(a) + abs(b)
    if denom == 0.0
        return zero(a)
    else
        return (a * abs(b) + abs(a) * b) / denom
    end
end

# MC limiter: Lim(a,b) = minmod(2a, (a+b)/2, 2b)
mc(a, b) = minmod(2a, 0.5 * (a + b), 2b)

# superbee limiter: Lim(a,b) = maxmod2( minmod2(2a,b), minmod2(a,2b) )
superbee(a, b) = maxmod2(minmod2(2a, b), minmod2(a, 2b))


#####################
# Slope functions for each limiter
#####################
function minmod_slope(left, center, right, theta)
    forward_diff  = right .- center
    backward_diff = center .- left
    central_diff  = (forward_diff .+ backward_diff) ./ 2.0
    return minmod.(theta .* forward_diff, central_diff, theta .* backward_diff)
end

function vanleer_slope(left, center, right)
    backward_diff = center .- left
    forward_diff  = right  .- center
    return vanleer.(backward_diff, forward_diff)
end

function mc_slope(left, center, right)
    backward_diff = center .- left
    forward_diff  = right  .- center
    return mc.(backward_diff, forward_diff)
end

function superbee_slope(left, center, right)
    backward_diff = center .- left
    forward_diff  = right  .- center
    return superbee.(backward_diff, forward_diff)
end


#############################
# Limiter → slope dispatch
#############################

slope(lim::MinmodLimiter, left, center, right) = minmod_slope(left, center, right, lim.theta)

slope(::VanLeerLimiter, left, center, right) = vanleer_slope(left, center, right)

slope(::MCLimiter, left, center, right) = mc_slope(left, center, right)

slope(::SuperbeeLimiter, left, center, right) = superbee_slope(left, center, right)