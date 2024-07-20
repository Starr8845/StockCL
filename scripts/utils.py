def string_format(x):
    out = ""
    if type(x)==dict:
        for key in x:
            if type(x[key])==list:
                out+= str(key)+': ' + ' '.join(['%.2e,'%(a) for a in x[key] ])
            else:
                out+=str(key)+ ': %.2e; '%(x[key])
    else:
        out = '%.2e'%(x)
    return out
