
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import enum 

def DigitalPayoffValuation(S,T,r,payoff): 
    return np.exp(-r*T) * np.mean(payoff(S))

def GeneratePathsGBMEuler(NoOfPaths,NoOfSteps,T,r,sigma,S_0):    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps+1])
   
    # Euler Approximation
    S1 = np.zeros([NoOfPaths, NoOfSteps+1])
    S1[:,0] =S_0
    
    time = np.zeros([NoOfSteps+1])
        
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):
        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
        
        S1[:,i+1] = S1[:,i] + r * S1[:,i]* dt + sigma * S1[:,i] * (W[:,i+1] - W[:,i])
        time[i+1] = time[i] +dt
        
    # Retun S1 and S2
    paths = {"time":time,"S":S1}
    return paths

def UpAndOutBarrier(S,T,r,payoff,Su):
        
    # handling of a barrier
    n1,n2 = S.shape
    barrier= np.zeros([n1,n2]) + Su  # barrier matrice de meme taille que S mais les valeurs mta3a lkol Su (li hia seuil mta3i)
    
    hitM = S > barrier                
    hitVec = np.sum(hitM, 1)        # traja3li list , somme içi sur les lignes kan ligne dima te7et Su rahy case mta3ha fiha 0
                                    # ou ligne lifatou yabda fihom entier >= 1 
        
    hitVec = (hitVec == 0.0).astype(int)  # list fiha 1 kan sénario dima te7et il Su et 0 kan fat 3al li9lila mara 
    
    V_0 = np.exp(-r*T) * np.mean(payoff(S[:,-1]*hitVec))  # valuation kan lil les lignes li dima te7et 
                                                          # les sénarios li fatou 7ata mara barka il Su bich ya3touni Payoff=0
    return V_0

def mainCalculation():
    NoOfPaths = 10000
    NoOfSteps = 250
   
    S0    = 100.0
    r     = 0.05
    T    = 5
    sigma = 0.2
    Su = 150
    
    paths = GeneratePathsGBMEuler(NoOfPaths,NoOfSteps,T,r,sigma,S0)
    S_paths= paths["S"]
    S_T = S_paths[:,-1]
    
    # Payoff setting    
    K  = 100.0
    K2 = 140.0  # Cap Si S>K2 alors le gain rest cte (pour que le gain ne diverge pas) 
    
    # Payoff specification
    payoff = lambda S: np.maximum(S-K,0.0) # payoff sans Cap
    payoff1 = lambda S: np.maximum(S-K,0.0) - np.maximum(S-K2,0) # payoff avec Cap quelque soit la valeur de S si S>k2 
                                                                 # alors le payoff rest constant  
    
    #Plot
    S_T_grid = np.linspace(50,S0*1.5,200)
    
    plt.figure(1)
    plt.plot(S_T_grid,payoff(S_T_grid), label='Sans Cap "PayOff"')
    plt.plot(S_T_grid,payoff1(S_T_grid), label='Avec Cap "PayOff1"')
    plt.xlabel('Stock price')
    plt.ylabel('Payoff')
    plt.legend()
    
    # Valuation Cas 1 : 
    val_t0 = DigitalPayoffValuation(S_T,T,r,payoff)
    print("Value of the contract at t0 ={0}".format(val_t0))
    
    # Valuation Cas 2 : avec une barrier "Cap" pour limiter le gain dans le cas où S augmente beaucoup
    barrier_price1 = DigitalPayoffValuation(S_T,T,r,payoff1)  
    print("Value of the barrier contract (Cas 1) at t0 ={0}".format(barrier_price1))
    # Le valeur de l'option içi est plus faible que le cas 1 car on a limiter le gain
    
    
    # barrier pricing Cas 3 :
    barrier_price2 = UpAndOutBarrier(S_paths,T,r,payoff,Su)  
    print("Value of the barrier contract (Cas 2) at t0 ={0}".format(barrier_price2))
    # najam nista3mal payoff1 içi zada ya3ni n7ebha dima te7et Su ou kan fatet K2 rendement yabda cte
    # fi l7ala hathy price mta3 l'option yan9as akthar mil cas 3
    barrier_price3 = UpAndOutBarrier(S_paths,T,r,payoff1,Su)
    print("Value of the barrier contract (Cas 3) at t0 ={0}".format(barrier_price3))
    
    print("\nREMARQUE: Prix Cas1 > Prix Cas2 > Prix Cas3")
mainCalculation()