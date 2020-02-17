import scipy.integrate as integ
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo

M_sun, R_sun=1.98919e33, 6.9599e10 #g, cm
X_sun, Z_sun=0.709723, .0202687
n_sun=3.19
L0_sun=3.846e33 #erg s-1
G=6.67e-11
##script file###

def lane(t,r,n):
    "define a equação a resolver"
    y1, y2 = r
    dy1= y2
    dy2=-np.absolute(y1)**n-2*y2/t
    return dy1, dy2
 
def zero(t,y):
    "função definida para encontrar o zero e parar a integração nesse valor, na resolução da equação de lane-emden através do modulo scipy.integrate.solve_ivp"
    return y[0]
zero.terminal=True

def sol_lane(n=n_sun,a=1e-10,max_step=.005,b=35):
    "resolve a equação de lane emden num intervalo [a,b], com y inicial y1=1-a²/2 e y2=-a, interrompe a integração ao encontrar um zero"
    sol=integ.solve_ivp(lambda t,r: lane(t,r,n), [a,b], \
    [1-a**2/2,-a], first_step=a, max_step=max_step,events=zero)
    sol0=np.array([0,1,0]) 
    sol1=np.array([sol.t,sol.y[0],sol.y[1]])
    sol1[1,-1]=1e-18 #define um valor pŕoximo de 0 para dtheta em xi_s, mas não igual, de modo a assegurar o progresso do código
    return np.concatenate((sol0.reshape(3,1), sol1), axis=1)
    #retorna [xi,theta,dtheta]

def grandezas(data,n=n_sun,M=M_sun,R=R_sun,X=X_sun,Z=Z_sun,G=6.67430e-8,gas_const=8.314462618e7):
    "Obtém rho, P, T para uma dada estrela. data=[xi, theta, dtheta]"
    mu=4/(5*X-Z+3)
    rho_0=data[0,-1]*M/(4*np.pi*R**3*-data[2,-1])
    P_0=(G*M**2)/(R**4*4*np.pi*(n+1)*data[2,-1]**2)
    T_0=(mu*G*M)/(R*gas_const*(n+1)*data[0,-1]*-data[2,-1])
    rho=rho_0*data[1]**n
    P=P_0*data[1]**(n+1)
    T=T_0*data[1]
    return rho,P,T

def emiss(rho,T,n=n_sun,M=M_sun,R=R_sun,X=X_sun,Z=Z_sun):
    T6=T*1e-6
    alpha=1.2e17*((1-X-Z)/(4*X))**2*np.exp(-100*T6**(-1/3))
    phi=1-alpha+np.sqrt(alpha*(alpha+2))
    F1=(np.sqrt(alpha+2)-np.sqrt(alpha)) / (np.sqrt(alpha+2)+3*np.sqrt(alpha))
    F2=(1-F1) / (1+8.94e15*(X/(4-3*X))*T6**(-1/6)*np.exp(-102.65*T6**(-1/3)))
    F3=1-F1-F2

    e_0=2.38e6*X**2*rho*T6**(-2/3) * (1+0.0123*T6**(1/3)+0.0109*T6**(2/3)\
    +0.00095*T6) * np.exp(-33.80*T6**(-1/3)+0.27*rho**(1/2)*T6**(-3/2))
    e_pp = e_0/0.980*phi * (0.98*F1+0.96*F2+0.721*F3)
    e_cno = 8.67e27*Z*X*rho*T6**(-2/3) * (1+.0027*T6**(1/3)-.00778*T6**(2/3)\
    -.000149*T6)*np.exp(-152.28*T6**(-1/3))
    return e_pp,e_cno

def f_lum(data,e,n=n_sun,M=M_sun,L0=L0_sun):
    "integra de modo a obter a luminosidade"
    return 1/(-data[0,-1]**2*data[2,-1])*integ.simps(data[0]**2*data[1]**n*M*e/L0,x=data[0])

def luminosidade(n=n_sun,M=M_sun,R=R_sun,X=X_sun,Z=Z_sun, L=L0_sun):
    "reune os params necessários para proceder à integração"
    lane=sol_lane(n)
    rho,P,T=grandezas(lane,n,M,R,X,Z)
    emis=emiss(rho,T,n,M,R,X,Z)
    e=emis[0]+emis[1]
    return f_lum(lane,e,n,M,L)

def get_n(M=M_sun,R=R_sun,Z=Z_sun,L=L0_sun,X=X_sun,k=(2.4,3.9)):
    "encontrar o zero de 1-Lr através do modulo root_scalar do scipy.optimize"
    zero=spo.root_scalar(lambda n: 1-luminosidade(n,M,R,X,Z,L),bracket=k)
    return zero.root

##método de monte carlo --> n
def erro(M,R,L,Z,X,dM,dR,dL, N,bracket):
    "gera N amostras de n com um desvio padrão dos params M, L e R, de modo a obter uma estimativa de valor médio de n bem como o seu próprio desvio padrão"
    ns=np.zeros(N)
    for i in range(N):
        Mr=M+dM*np.random.normal()
        Rr=R+dR*np.random.normal()
        Lr=L+dL*np.random.normal()
        ns[i]=get_n(Mr,Rr,Z,L,X,bracket)
    return ns

##massas

def massa(tab):
    "usa a relação 3.40 da sebenta para obter m(r)"
    return (tab[0]/tab[0,-1])**2*(tab[2]/tab[2,-1])

def lumi_var(data,M,L0,e,n):
    L=np.zeros((e.size))
    for i in range(2,e.size):
        L[i]=1/(-data[0,-1]**2*data[2,-1])*integ.simps(data[0,:i]**2*data[1,:i]**n*M*e[:i]/L0,x=data[0,:i])
    return L


def valid_2n(M,Ms,xi,R):
    "obtém n por método alternativo"
    V=integ.simps(G*Ms[2:]/xi[2:],x=Ms[2:])
    V1=G*Ms[-1]**2/xi[-1]
    return 3*(V1/V)-5
