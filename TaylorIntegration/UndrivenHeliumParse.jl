#Módulo con ecuaciones de movimiento, hamiltoniano y test de energía de la configuración Zee

#__precompile__(true)


#module UH
const Z = 2.0
#Pkg.checkout("TaylorIntegration", "parse_eqs")
#Pkg.checkout("TaylorSeries", "mutating_functions")
using TaylorSeries, TaylorIntegration
#Regularización KS
f(x, y) = [x^2 - y^2, 2x*y]
f(v) = f(v...)

#Ecuaciones de movimiento 1D
function undrivenHelium1D!(τ, q, dq)
    Q₁, Q₂, P₁, P₂ = q
    
    t1 = Q₁^2
    t2 = Q₂^2
    t = t1 * t2
    #dq[1] = t
    R12 = t1 - t2
    aR12 = abs(R12)
    RRR = aR12^3
    c1 = R12/RRR
    f1 = (1 + 1/aR12)
    f2 = t*c1
   
    dq[1] = 0.25*t2*P₁
    dq[3] = 2*Q₁*(-0.125*P₂^2 + Z  - t2*f1 + f2)
    dq[2] = 0.25*t1*P₂
    dq[4] = 2*Q₂*(-0.125*P₁^2 + Z  - t1*f1 - f2)

#    return [t, q₁, q₂, p₁, p₂] 
    nothing
end



#Test de compatibilidad eom-Hamiltoniano
function errHam2D(N::Int)
    srand(487293456)
    J = vcat(  hcat(zeros(4,4), eye(4,4)), hcat(-eye(4,4), zeros(4,4))  )
    
    var2D = set_variables("q₁x q₁y q₂x q₂y p₁x p₁y p₂x p₂y", order = 1)
    
    dnorm = zeros(N)
    
    for j in 1:N
        al = condini2D(rand(5)...)
        #al = rand(8)
        meq = similar(al)
        alt = al + var2D
        ene = J*∇(regHam2D(alt))
        ene1 = Float64[ene[k].coeffs[1].coeffs[1] for k in 1:8]
        undrivenHelium2D!(0.0, al, meq)
        dnorm[j] = norm(meq - ene1)/eps()
        return dnorm
    end
end



#Ecuaciones de movimiento 2D
function undrivenHelium2D!{T<:Number}(τ, q::Array{T,1}, dq::Array{T,1})
    Q₁x, Q₁y, Q₂x, Q₂y, P₁x, P₁y, P₂x, P₂y = q
    
    #Cantidades auxiliares
    Q₁² = Q₁x^2 + Q₁y^2
    Q₂² = Q₂x^2 + Q₂y^2
    P₁² = P₁x^2 + P₁y^2
    P₂² = P₂x^2 + P₂y^2
    t = Q₁²*Q₂²
    rf = f(Q₁x, Q₁y) - f(Q₂x, Q₂y)
    f₁, f₂ = rf
    fs = f₁^2 + f₂^2
    nf = sqrt(fs)
    c1 = 1 + 1/nf
    nf³ = nf^3
    factor1 = t/nf³
    s1 = -0.125P₂² + Z - Q₂²*c1
    s2 = factor1*f₁
    s3 = -0.125P₁² + Z - Q₁²*c1
    #@show c1
    dq[1], dq[2] = 0.25*Q₂²*[P₁x, P₁y]
    dq[3], dq[4] = 0.25*Q₁²*[P₂x, P₂y]
    dq[5] = 2*Q₁x*(s1 + s2) + factor1*f₂*Q₁y
    dq[6] = 2*Q₁y*(s1 - s2) + factor1*f₂*Q₁x
    dq[7] = 2*Q₂x*(s3 - s2) - factor1*f₂*Q₂y
    dq[8] = 2*Q₂y*(s3 + s2) - factor1*f₂*Q₂x
    nothing
end

#Variables para tests de compatibilidad hamiltoniano eom


#Condiciones Iniciales en 1D
function condini1D{T<:Number}(x10::T, px10::T)
    @assert x10 != 0
    Q1 = sqrt(x10)
    Q2 = 0.0
    P1 = 2*px10*sqrt(x10)    
    P2 = sqrt(8Z)    
    return [Q1, Q2, P1, P2]
end

#Condiciones Iniciales 2D
function condini2D{T<:Number}(q₁x::T, q₁y::T, p₁x::T, p₁y::T, P₂y::T)
    @assert (q₁x > 0 || q₁y > 0) && P₂y^2 <= 8Z
    Q₁x = sqrt(sqrt(4q₁x^2 + q₁y^2) + 2q₁x)/2
    Q₁y = q₁y/(2Q₁x)
    Q₂x = 0.0
    Q₂y = 0.0
    P₁x = 2(Q₁x*p₁x + Q₁y*p₁y)
    P₁y = 2(Q₁x*p₁y - Q₁y*p₁x)
    P₂x = sqrt(8Z - P₂y^2)
    return T[Q₁x, Q₁y, Q₂x, Q₂y, P₁x, P₁y, P₂x, P₂y]
end




#Hamiltoniano en coord. regularizadas 2D
function regHam2D(q₁x, q₁y, q₂x, q₂y, p₁x, p₁y, p₂x, p₂y)
    #Cantidades auxiliares
    Q₁² = q₁x^2 + q₁y^2
    Q₂² = q₂x^2 + q₂y^2
    P₁² = p₁x^2 + p₁y^2
    P₂² = p₂x^2 + p₂y^2
    t = Q₁²*Q₂²
    rf = f(q₁x, q₁y) - f(q₂x, q₂y)
    f₁, f₂ = rf
    nf = (f₁^2 + f₂^2)^(1/2)
    
    H = 0.125*(P₁²*Q₂² + P₂²*Q₁²) - Z*(Q₁² + Q₂²) + t*(1.0 + 1.0/nf)
    return H
end



regHam2D(v) = regHam2D(v...)

#Hamiltoniano en coord. regularizadas 1D
function regHam1D(Q₁, Q₂, P₁, P₂)
    #Cantidades auxiliares
    P₁² = P₁^2
    P₂² = P₂^2
    Q₁² = Q₁^2
    Q₂² = Q₂^2
    nf = abs(Q₁² - Q₂²)
    
    H = 0.125*(P₁²*Q₂² + P₂²*Q₁²) - Z*(Q₁² + Q₂²) + Q₁²*Q₂²*(1.0 + 1.0/nf)
    return H
end

regHam1D(v) = regHam1D(v...)

#end



function errHam1D(N::Int)
    srand(487293456)
    J = vcat(  hcat(zeros(2,2), eye(2,2)), hcat(-eye(2,2), zeros(2,2))  )
    
    var1D = set_variables("q₁ q₂ p₁ p₂", order = 1)
    
    dnorm = zeros(N)
    als = typeof(zeros(4))[]

    for j in 1:N
        al = condini1D(rand(2)...)
        meq = similar(al)
        #al = [BigFloat(x) for x in al1]
        alt = al + var1D
        ene = J*∇(regHam1D(alt))
        ene1 = [ene[k].coeffs[1].coeffs[1] for k in 1:4]
        undrivenHelium1D!(0.0, al, meq)
        push!(als, al)
        meq[1] = 0
        dnorm[j] = norm(meq - ene1)/eps() 
    end
    return dnorm, als 
end