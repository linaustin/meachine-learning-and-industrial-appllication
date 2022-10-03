# %%
import cmath

def solve_2ndorder_equation(a,b,c):
    root_sqrt = cmath.sqrt(b**2 - (4*a*c))

    root1_real = (-b + root_sqrt.real)/(2*a)
    root1_imag = root_sqrt.imag/(2*a)

    root2_real = (-b - root_sqrt.real)/(2*a)
    root2_imag = -1*(root_sqrt.imag/(2*a))

    root1 = complex(root1_real, root1_imag)
    root2 = complex(root2_real, root2_imag)

    print(f"root1 : {root1} root2: {root2}")

# %%

print("solve 2nd polynomial (a*x^2 + b*x +c = 0")

a = int(input("enter coeficent of x^2\n"))
b = int(input("enter coeficent of x\n"))
c = int(input("enter constant\n"))

solve_2ndorder_equation(a,b,c)


