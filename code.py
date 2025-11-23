import sympy as sp

x,y, t =  sp.symbols('x y t')

u = sp.Funtion('u')(x,y,t)

pde = sp.Eq(sp.diff(u,x) 
