#Rutinas de integración de ecuaciones variacionales usando TaylorIntegration

__precompile__(true)

using TaylorSeries, TaylorIntegration


#Métodos para integración de ecuaciones variacionales y tests de simplecticidad
function var_jetcoeffs!{T<:Number}(eqsdiff!, t0::T, x::Vector{Taylor1{T}},
        dx::Vector{Taylor1{T}}, xaux::Vector{Taylor1{T}}, jx::Vector{Taylor1{T}},
        jdx::Vector{Taylor1{T}}, jxaux::Vector{Taylor1{T}},
        δx::Array{TaylorN{Taylor1{T}},1}, dδx::Array{TaylorN{Taylor1{T}},1},
        jac::Array{Taylor1{T},2}, vT::Vector{T})

    order = x[1].order
    vT[1] = t0

    # Dimensions of phase-space: dof
    dof = length(x)

    for ord in 1:order
        ordnext = ord+1

        # Set `xaux`, auxiliary vector of Taylor1 to order `ord`
        for j in eachindex(x)
            @inbounds xaux[j] = Taylor1( x[j].coeffs[1:ord] )
        end
        
        #Set `jxaux` 
        for j in eachindex(jx)
            @inbounds jxaux[j] = Taylor1( jx[j].coeffs[1:ord] ) 
        end
        
        # Equations of motion
        tT = Taylor1(vT[1:ord])
        eqsdiff!(tT, xaux, dx)
        
        # stabilitymatrix!( eqsdiff!, t0, xaux[1:dof], δx, dδx, jac )
        TaylorIntegration.stabilitymatrix!( eqsdiff!, t0, xaux, δx, dδx, jac )
        @inbounds jdx[1:end] = jac * reshape( jxaux[1:end], (dof,dof) )
        
        # Recursion relations
        for j in eachindex(x)
            @inbounds x[j][ordnext] = dx[j][ord]/ord
        end
        for j in eachindex(jx)
            @inbounds jx[j][ordnext] = jdx[j][ord]/ord
        end
        
    end
    nothing
end

function var_taylorstep!{T<:Number}(f!, x::Vector{Taylor1{T}}, dx::Vector{Taylor1{T}},
        xaux::Vector{Taylor1{T}}, jx::Vector{Taylor1{T}}, jdx::Vector{Taylor1{T}},
        jxaux::Vector{Taylor1{T}}, δx::Array{TaylorN{Taylor1{T}},1},
    dδx::Array{TaylorN{Taylor1{T}},1}, jac::Array{Taylor1{T},2}, t0::T, t1::T, x0::Array{T,1}, 
        arrjac::Array{T,1}, order::Int, abstol::T, vT::Vector{T})
    
    # Compute the Taylor coefficients
    var_jetcoeffs!(f!, t0, x, dx, xaux, jx, jdx, jxaux, δx, dδx, jac, vT)
    
    # Compute the step-size of the integration using `abstol`
    δt = TaylorIntegration.stepsize(x, abstol)
    δt = min(δt, t1-t0)
    
    
    # Update x0
    evaluate!(x, δt, x0)
    
    #Update jt
    evaluate!(jx, δt, arrjac)
    
    return δt
end


#Versión de ecuaciones variacionales
function vartaylorinteg{T<:Number}(f!, q0::Array{T,1}, t0::T, tmax::T,
        order::Int, abstol::T; maxsteps::Int=500)

    # Allocation
    const tv = Array{T}(maxsteps+1)
    dof = length(q0)
    const xv = Array{T}(dof, maxsteps+1)
    const jt = eye(T, dof, dof)
    const mv = Array{Array{T,2}}(maxsteps+1)
    const vT = zeros(T, order+1)
    vT[2] = one(T)

    
    # NOTE: This changes GLOBALLY internal parameters of TaylorN
    global _δv = set_variables("δ", order=1, numvars=dof)
    
    # Initialize the vector of Taylor1 expansions
    const x = Array{Taylor1{T}}(dof)
    const dx = Array{Taylor1{T}}(dof)
    const xaux = Array{Taylor1{T}}(dof)
    
    #Taylor1 for jacobian
    const jx = Array{Taylor1{T}}(dof*dof)
    const jdx = Array{Taylor1{T}}(dof*dof)
    const jxaux = Array{Taylor1{T}}(dof*dof)
    const δx = Array{TaylorN{Taylor1{T}}}(dof)
    const dδx = Array{TaylorN{Taylor1{T}}}(dof)
    const jac = Array{Taylor1{T}}(dof, dof)
    const arrjac = Array{T}(dof*dof)
    for i in eachindex(q0)
        @inbounds x[i] = Taylor1( q0[i], order )
    end
    
    #auxiliary arrays for symplectic structure tests
    const δSv = Array{T}(maxsteps+1); δSv[1] = zero(T)
    auxJn = Int(dof/2)
    const J_n = vcat(  hcat(zeros(auxJn,auxJn), eye(auxJn,auxJn)), hcat(-eye(auxJn,auxJn), zeros(auxJn,auxJn))  )

    # Initial conditions
    @inbounds tv[1] = t0
    @inbounds xv[:,1] .= q0
    @inbounds mv[1] = jt
    x0 = copy(q0)
    arrjac = reshape(jt, dof*dof)
    
    for i in eachindex(arrjac)
        @inbounds jx[i] = Taylor1( arrjac[i], order )
    end
    
    # Integration
    nsteps = 1
    while t0 < tmax
        δt = var_taylorstep!(f!, x, dx, xaux, jx, jdx, jxaux, δx, dδx, jac, t0, tmax, x0, 
        arrjac, order, abstol, vT)
        
        #Taking stability matrix values
        for ind in eachindex(jt)
            @inbounds jt[ind] = arrjac[ind]
        end
        
        t0 += δt
        nsteps += 1
        @inbounds tv[nsteps] = t0
        
        
        
        δSv[nsteps] = norm( jt'*J_n*jt-J_n, Inf)
        
        for i in eachindex(x0)
            @inbounds x[i][1] = x0[i]
        end
        
        for i in eachindex(arrjac)
            @inbounds jx[i][1] = arrjac[i]
        end

        @inbounds xv[:,nsteps] .= x0
        if nsteps > maxsteps
            warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end

    return view(tv,1:nsteps), view(transpose(xv),1:nsteps,:), δSv, mv
end