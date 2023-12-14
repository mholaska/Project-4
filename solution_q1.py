P_B = {"+b": 0.001,
       "-b": 0.999}

P_E = {"+e": 0.002,
       "-e": 0.998}

P_JgivenA = {"+a+j":0.9,
        "+a-j":0.1,
        "-a+j":0.05,
        "-a-j":0.95}

P_MgivenA = {"+a+m":0.7,
        "+a-m":0.3,
        "-a+m":0.01,
        "-a-m":0.99}

P_AgivenBE = {"+b+e+a":0.95,
         "+b+e-a":0.05,
         "+b-e+a":0.94,
         "+b-e-a":0.06,
         "-b+e+a":0.29,
         "-b+e-a":0.71,
         "-b-e+a":0.001,
         "-b-e-a":0.999,}

A = ["+a","-a"]
J = ["+j","-j"]
B = ["+b","-b"]
E = ["+e","-e"]

# Given the above probabilities we can perform variable elimination to find P(B|+j)


# We found the marginal distribution to be the sum over e & a of P(B,j,e,a) from the Bayes net

def joinOverA():
    table = {}
    for i in A:
        for j in J:
            JgivenA_val = P_JgivenA[i+j]
            for k in B:
                for l in E:
                    table[i+j+k+l] = JgivenA_val*P_AgivenBE[k+l+i]
    return table

def sumOutA():
    table = {}
    for j in J:
        for k in B:
            for l in E:
                table[j+k+l] = P_JAgivenBE["-a"+j+k+l]+P_JAgivenBE["+a"+j+k+l]
    return table
                
def joinOverE():
    table = {}
    for j in J:
        for k in B:
            for l in E:
                table[j+l+k] = P_E[l]*P_JgivenBE[j+k+l]
    return table

def sumOutE():
    table = {}
    for j in J:
        for k in B:
            table[j+k] = P_JEgivenB[j+"+e"+k]+P_JEgivenB[j+"-e"+k]
    return table

def joinOverB():
    table = {}
    for j in J:
        for k in B:
                table[j+k] = P_B[k]*P_JgivenB[j+k]
    return table

def normalizeJB():
    Z = 1 / (P_JB["+j+b"]+P_JB["+j-b"])
    table = {
        "+j+b": P_JB["+j+b"]*Z,
        "+j-b": P_JB["+j-b"]*Z
    }
    return table

P_JAgivenBE = joinOverA()

P_JgivenBE = sumOutA()

P_JEgivenB = joinOverE()

P_JgivenB = sumOutE()

P_JB = joinOverB()

P_BgivenplusJ = normalizeJB()

print("All Computed Distributions:")
print("\nP(JA|BE)")
for i in P_JAgivenBE:
    print(i, round(P_JAgivenBE[i],4))

print("\nP(J|BE)")
for i in P_JgivenBE:
    print(i, round(P_JgivenBE[i],4))

print("\nP(JE|B)")
for i in P_JEgivenB:
    print(i, round(P_JEgivenB[i],4))

print("\nP(J|B)")
for i in P_JgivenB:
    print(i, round(P_JgivenB[i],4))

print("\nP(JB)")
for i in P_JB:
    print(i, round(P_JB[i],4))

print("\nP(B | +j) (Final Distribution)")
for i in P_BgivenplusJ:
    print(i, round(P_BgivenplusJ[i],4))