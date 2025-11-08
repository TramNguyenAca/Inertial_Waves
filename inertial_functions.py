import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as lin
import scipy.integrate as intg
import matplotlib.pyplot as plt

##### Finite Deference discretization #####
class FiniteDifferenceSphereFourierlong:# Spherical coordinate, Fourier transfrom in longtitude
                                        # [theta, m]=[latitute, Fourier(longtitude)]
    def __init__(self, x, dx, nx, omega_ref, r = 1, **kwargs):  
        self.dx = dx
        self.nx = nx
        self.omega_ref = omega_ref
        
        self.Laplace = None
        self.Laplace_homo = None

        ### part 1
        x_back  = x - dx/2 # x at shifted grid points back/forward
        x_forth = x + dx/2
        
        # cos, sin,tan r^2sin functions        
        self.sin,self.cos,self.tan = (np.sin(x),np.cos(x),np.tan(x)) 
        self.r2sin                 = r**2*self.sin    
        sin_back, sin_forth        = (np.sin(x_back),np.sin(x_forth)) 
        
        ### part 2: Cartesian 1st, 2nd Differentiation
        # Diff = d/dthe                       #center scheme, one side scheme boundaries
        D = np.eye(nx, k=1) - np.eye(nx, k=-1)
        D[0,0],D[0,1],D[0,2]                   = (-3,4,-1)          
        D[nx-1,nx-3],D[nx-1,nx-2],D[nx-1,nx-1] = (1,-4,3)
        self.Diff = 1/(2*dx)*sp.csr_matrix(D)         
        #print(self.Alpha.toarray())
        
        # DiffSecond = d^2/dthe^2              #center scheme, one side scheme boundaries
        D = np.eye(nx,k=-1) -2*np.eye(nx) + np.eye(nx,k=1) 
        D[0,0],D[0,1],D[0,2],D[0,3]                         = (2,-5,4,-1)          
        D[nx-1,nx-4],D[nx-1,nx-3],D[nx-1,nx-2],D[nx-1,nx-1] = (-1,4,-5,2)        
        self.DiffSecond = 1/dx**2*sp.csr_matrix(D) 
        
        ### part 3: Spherical Laplacian, Gradient. Do not include 1/r^2sin!
        ### Compute L_theta first, add L_m later        
        # Compute Laplace_homo = L_the_homo + L_m          # zero boundary 
        self.L_the_homo = 1/dx**2*(sp.diags(sin_back[1:],-1) -sp.diags((sin_back+sin_forth)) + sp.diags(sin_forth[:-1],1) )
        
        # Compute Laplace = L_the + L_m                    # no boundary
        # Compute L_the = (sin.phi')' = sin.phi'' + cos.phi'
        self.L_the = sp.diags(self.sin)@self.DiffSecond + sp.diags(self.cos)@self.Diff    
        
        
        '''
        #### Higher order derivatives
        D = np.eye(nx,k=-2)-8*np.eye(nx,k=-1)+8*np.eye(nx,k=1)-np.eye(nx,k=2)
        D = D/6
        D[0,0],D[0,1],D[0,2]                   = (-3,4,-1)          
        D[nx-1,nx-3],D[nx-1,nx-2],D[nx-1,nx-1] = (1,-4,3)
        self.Diff = 1/(2*dx)*sp.csr_matrix(D) 
        
        D = -np.eye(nx,k=-2)+16*np.eye(nx,k=-1)-30*np.eye(nx)+16*np.eye(nx,k=1)-np.eye(nx,k=2)
        D = D/12
        D[0,0],D[0,1],D[0,2],D[0,3]                         = (2,-5,4,-1)          
        D[nx-1,nx-4],D[nx-1,nx-3],D[nx-1,nx-2],D[nx-1,nx-1] = (-1,4,-5,2) 
        self.DiffSecond = 1/dx**2*sp.csr_matrix(D) 
        
        self.L_the = sp.diags(self.sin)@self.DiffSecond + sp.diags(self.cos)@self.Diff  
        '''
        
        # Isomorphism in H1, H2 with Dirichlet boundary; L_the only, not include L_m!
        # Rexeompute Diff_D, Diffsecond_D
        D = np.eye(nx, k=1) - np.eye(nx, k=-1)
        Diff_D = 1/(2*dx)*sp.csr_matrix(D) 
        D = np.eye(nx,k=-1) -2*np.eye(nx) + np.eye(nx,k=1)  
        DiffSecond_D = 1/dx**2*sp.csr_matrix(D)
        
        L_the_D   = sp.diags(self.sin)@DiffSecond_D+ sp.diags(self.cos)@Diff_D  

        LaplaceIso_D  = L_the_D- sp.diags(self.sin)  - sp.diags(1/self.sin) 
                   
        self.Iso1 = -LaplaceIso_D   
        self.Iso2 = LaplaceIso_D@(sp.diags(1/self.r2sin)@LaplaceIso_D) 
        
        #Isomorphism in H1, H2 with Neumann boundary. Iso = (-L_the_N +Id)^{-1}
        # Recompute Diffsecond_N
        D = np.eye(nx,k=-1) -2*np.eye(nx) + np.eye(nx,k=1)  
        D[0,0],D[nx-1,nx-1]       = (-1,-1)      # Neumann bc in second Diff          
        DiffSecond_N = 1/dx**2*sp.csr_matrix(D)
        
        L_the_N   = sp.diags(self.sin)@DiffSecond_N + sp.diags(self.cos)@self.Diff  
        LaplaceIso_N  = L_the_N - sp.diags(self.sin) 
      
        self.Iso1_N = -LaplaceIso_N 
        self.Iso2_N = LaplaceIso_N@(sp.diags(1/self.r2sin)@LaplaceIso_N) 

    def Wave(self,w,m,gamma,omega):        
        
        # Laplace = L_the + L_m. Do not include 1/r^2sin => multiply by r^2sin when solving PDE  
        L_m   = -m**2*sp.diags(1/self.sin)
                    
        self.Laplace_homo = self.L_the_homo + L_m 
        self.Laplace      = self.L_the      + L_m       
         
        
        # Bi_Laplace: divide by r^2sin before taking 2nd Laplace
        self.Bi_Laplace = self.Laplace@(sp.diags(1/self.r2sin)@self.Laplace_homo) 
               
        # Alpha term       
        diff_omega = self.Diff@omega
        alpha = self.L_the@omega + 2*(-self.sin*omega + self.cos*diff_omega )
        Alpha = sp.diags(alpha)

        ### Primal equation. Beta_term = bLap(phi)
        Wave       = -gamma*self.Bi_Laplace - 1j*w*self.Laplace_homo - 1j*m*Alpha
        
        beta = omega - self.omega_ref 
        Beta_term  = sp.diags(beta)@self.Laplace_homo 
        
        WavePrimal = Wave + 1j*m*Beta_term    
        
        
        ### Adjoint equation. Beta term = Lap(bz) 
        # Beta term = bLap(z) + L_the(b)z + 2*diff(b).diff(z)  !!! Important: L_the is not Lap
        Beta_term = Beta_term+sp.diags(self.L_the@beta) +2*sp.diags(self.sin*diff_omega)@self.Diff 
        
        WaveAdjoint= (Wave + 1j*m*Beta_term).conjugate()        
        
        return WavePrimal,WaveAdjoint 
        

# PDE solution, adjoint calculation, isomorphism calculation, observation operators
class PartialDiffEq:
    def __init__(self, x, dx, nx, source, r, m, FD, leak_level_real, leak_level_imag, **kwargs):
        #self.__dict__.update(FD.__dict__)
        self.r2sin       = FD.r2sin
        self.tan         = FD.tan
        
        self.FD = FD
        self.dx = dx
        self.nx = nx
        self.source = source
        self.r = r
        self.m = m
        
        self.lu_Iso1   = lin.splu(sp.csc_matrix(FD.Iso1))
        self.lu_Iso2   = lin.splu(sp.csc_matrix(FD.Iso2))
        self.lu_Iso1_N = lin.splu(sp.csc_matrix(FD.Iso1_N))
        self.lu_Iso2_N = lin.splu(sp.csc_matrix(FD.Iso2_N))
        
        self.leak_index_real,self.leak_index_imag  = [int(leak_level_real/200*self.nx), int(leak_level_imag/200*self.nx)]
            
    ###	part 1. Note: multiply by (r^2sin)^2 to source when solving PDE  
    def Update(self, data, w, m, gamma, omega):        	      # run this first to update Wave operator   
        
        FD = self.FD
        
        # MUST RUN THIS FIRST to update wave operators!
        WavePA    = FD.Wave(w,m,gamma,omega)   
        
        # Update parameter-dependent Laplacian, convert to csc for LU decomp
        self.WavePrimal   = sp.csc_matrix(WavePA[0])
        self.WaveAdjoint  = sp.csc_matrix(WavePA[1])
        
        self.Laplace      = FD.Laplace
        self.Laplace_homo = FD.Laplace_homo
        self.Bi_Laplace   = FD.Bi_Laplace
     
        
        # LU decomposition for primal/adjoint operators
        self.lu_WavePrimal  = lin.splu(self.WavePrimal)    
        self.lu_WaveAdjoint = lin.splu(self.WaveAdjoint) 
               
        
        # Compute state, adjoint state, residual
        self.State    = self.lu_WavePrimal.solve(self.source*self.r2sin)  # State: = (Wave)^{-1}source
        
        if data.ndim == 2:    # cov data
            
            Gcovf     = self.lu_WavePrimal.solve((covf.T*self.r2sin).T)
            covstate  = self.lu_WavePrimal.solve((Gcovf.conj()*self.r2sin).T).T.conj()
            self.CovState = covstate
            
            self.Residual = self.CovState - data      
            
        else:                 # linear data with leakage option
            
            self.Residual = self.Leakage(self.State - data)  
                            
            
        self.AdjRes = self.lu_WaveAdjoint.solve((self.Residual.T*self.r2sin).T) # Adjoint of Res=ResCov

        
    def I(self,h,space,boundary):        	      # Isomorphism I(h)
        
        if space =='L2':    out = h
        if space =='H1' and boundary == 'Dirichlet':    out = self.lu_Iso1.solve(h*self.r2sin) 
        if space =='H1' and boundary == 'Neumann':      out = self.lu_Iso1_N.solve(h*self.r2sin) 
        if space =='H2' and boundary == 'Dirichlet':    out = self.lu_Iso2.solve(h*self.r2sin)   
        if space =='H2' and boundary == 'Neumann':      out = self.lu_Iso2_N.solve(h*self.r2sin)  
    
        return out
    
    def IntWeight(self,r):                # r^2sin Weighted Integral
        out = intg.trapz(r*self.r2sin,dx = self.dx)          
        
        return out
    
    ###	part 2   
    def Adjoint(self, space, boundary, data_dim):           # compute Adjoints   
        
        if data_dim == 2:
            phibar = (self.CovState).conj()   # cov data  
        else:
            phibar = self.State.conj()        # linear data 
            
        z = self.AdjRes  
        phiz =  phibar*z
        
        # Adjoint_gamma
        Adj_gamma = ((self.Bi_Laplace@phibar).T/self.r2sin).T*z 
        
        Adj_gamma = self.IntWeight(Adj_gamma.real) 
        
        if data_dim == 2:                     # cov data
            Adj_gamma = 2*self.IntWeight(Adj_gamma.real)
            
        
        # Adjoint_omega      
        Adj_omega  = 1/self.r**2*( self.FD.DiffSecond@phiz - (self.tan*(self.FD.Diff@phiz).T).T )                    
        Adj_omega  = Adj_omega - ((self.FD.Laplace_homo@phibar).T/self.r2sin).T*z
        Adj_omega  = Adj_omega.imag
        
        if data_dim == 2:                     # cov data
            Adj_omega = 2*intg.trapz(self.r2sin*Adj_omega, dx = self.dx, axis=1)       
            
        Adj_omega  = self.m*self.I(Adj_omega,space,boundary)
                                
        out = [Adj_gamma,Adj_omega]
        
        return out
    
        
    def Norm(self,u):   # L^2 error, of course can measure in H^p if necessary
        if u.ndim == 2:
            out = self.IntWeight(self.IntWeight(np.abs(u)**2))**1/2
        else:
            out =  self.IntWeight(np.abs(u)**2)**(1/2)
        return out 
    
    ### part 3: Observation operator = Leakage/restriction operator
    def Leakage(self,u):   
        
        if self.leak_index_real > 0:
            u[:self.leak_index_real].real  = 0
            u[-self.leak_index_real:].real = 0
            
        if self.leak_index_imag > 0:
            u[:self.leak_index_imag].imag   = 0
            u[-self.leak_index_imag:].imag  = 0
        
        return u
        
class GenerateData:   
    
    def __init__(self, PDE, data, noise_level):
        
        self.PDE = PDE
           
        noise = np.random.normal(0, 1, size= data.size)   + 1j*np.random.normal(0, 1, size= data.size) 
        self.noise = (noise_level/100)*(noise/PDE.Norm(noise))*PDE.Norm(data)
        self.data = data
        
    def Observe(self, data_type, noise_type):
        if noise_type.casefold() == 'clean'.casefold(): 
            out = self.data
        elif noise_type.casefold() == 'noisy'.casefold():
            out = self.data + self.noise
        else:
            raise ValueError("Noise type must be clean or noisy!")
        
        if data_type  == 'cov':
            out = p.outer(out, out.conj())
        
        return out
    
class PrintResult:
    def __init__(self, data_ext, data, omega_ext, gamma_ext, x, dx, nx, source, r, m, leak_level_real, leak_level_imag, **kwargs):
        
        self.data_ext = data_ext
        self.data = data
        
        self.omega_ext = omega_ext
        self.gamma_ext = gamma_ext        
        
        self.x = x
        self.dx = dx
        self.nx = nx
        
        self.leak_level_real = leak_level_real
        self.leak_level_imag = leak_level_imag
        
    def PrintLinear(self, \
                      gamma, omega, state, \
                      gamma_init, omega_init, state_init, \
                      residual, error_gamma, error_omega):
    
        data_ext = self.data_ext
        data = self.data
        
        omega_ext = self.omega_ext
        gamma_ext = self.gamma_ext 
        
        x = self.x
        dx = self.dx
        nx = self.nx
        
        leak_level_real = self.leak_level_real
        leak_level_imag = self.leak_level_imag
        
        errorstyle = 'r'
        initstyle  = 'k-'
        recostyle  = 'm.'
        exactstyle = 'c-'
        markersize = 2
        
        font = {'size'   : 15}
        tick_pos= [0, np.pi/2 , np.pi]
        labels = ['0', '$\pi$/2','$\pi$']
        
        rmax,rmin = [np.max([data_ext.real,state_init.real,state.real]),np.min([data_ext.real,state_init.real,state.real])]
        imax,imin = [np.max([data_ext.imag,state_init.imag,state.imag]),np.min([data_ext.imag,state_init.imag,state.imag])]
        leak_index_real = int(leak_level_real/200*nx)
        leak_index_imag = int(leak_level_imag/200*nx)    
        
        plt.figure(figsize=(8,7), dpi=150,constrained_layout=True)
        plt.subplot(331)
        plt.plot(x,gamma_ext*np.ones(nx),exactstyle,label="true");plt.plot(x,gamma*np.ones(nx),recostyle,markersize=markersize,label="reco")
        plt.plot(x,gamma_init*np.ones(nx),initstyle,alpha=0.2,label="init");
        plt.title('viscosity $\gamma$');      plt.xticks(tick_pos, labels); 
        plt.legend()

        plt.subplot(332)
        plt.plot(error_gamma,errorstyle); plt.title('relative error of $\gamma$'); #plt.yscale("log")   
        

        plt.subplot(334)
        plt.plot(x,omega_ext,exactstyle,label="true");plt.plot(x,omega,recostyle,markersize=markersize,label="reco");
        plt.title('rotation $\\Omega$')
        plt.plot(x,omega_init,initstyle,alpha=0.2,label="init");  plt.xticks(tick_pos, labels)
        plt.legend()

        plt.subplot(335)
        plt.plot(error_omega,errorstyle);plt.title('relative error of $\Omega$'); #plt.yscale("log")   
        
    
        plt.subplot(339)
        plt.plot(residual,errorstyle);plt.title('residual'); plt.xlabel('iterations',loc='right'); plt.yscale("log")   
        
 
        plt.subplot(337)
        plt.plot(x[leak_index_real:nx-leak_index_real],data[leak_index_real:nx-leak_index_real].real,exactstyle,label="data")
        plt.plot(x,state.real,recostyle,markersize=markersize,label="reco")
        plt.plot(x,state_init.real,initstyle,alpha=0.2,label="init");plt.title('$\Re(\Psi)$'); plt.xticks(tick_pos, labels)        

        if (leak_level_real > 0):            
            plt.fill_between(x,rmin,rmax, where= x<=   leak_level_real/2*dx, facecolor='green', alpha=0.2)
            plt.fill_between(x,rmin,rmax, where= x>=x[-1]-leak_level_real/2*dx, facecolor='green', alpha=0.2)
            
        plt.xlabel('latitude',loc='right')
        
    
        plt.subplot(338)
        plt.plot(x[leak_index_imag:nx-leak_index_imag],data[leak_index_imag:nx-leak_index_imag].imag,exactstyle,label="data")
        plt.plot(x,state.imag,recostyle,markersize=markersize,label="reco")
        plt.plot(x,state_init.imag,initstyle,alpha=0.2,label="init");plt.title('$\Im(\Psi)$'); plt.xticks(tick_pos, labels)
        
        if (leak_level_imag > 0) and (leak_level_imag < 100):
            plt.fill_between(x,imin,imax, where= x<=   leak_level_imag/2*dx, facecolor='green', alpha=0.2)
            plt.fill_between(x,imin,imax, where= x>=x[-1]-leak_level_imag/2*dx, facecolor='green', alpha=0.2)
            
        if (leak_level_imag == 100):
            plt.fill_between(x,imin,imax, where= x<= x[-1]  , facecolor='green', alpha=0.2)  
            
        plt.legend()
        
       
        plt.rc('font', **font); 
