import numpy as np
def compute_mi(x, y):
    mi = 0.0
    x_value_list = np.unique(x)
    y_value_list = np.unique(y)
    Px = np.array([ len(x[x==xval])/float(len(x)) for xval in x_value_list ]) #P(x)
    Py = np.array([ len(y[y==yval])/float(len(y)) for yval in y_value_list ]) #P(y)
    try:
        for i in range(len(x_value_list)):
            if Px[i] ==0.:
                continue
            sy = y[x == x_value_list[i]]
            if len(sy)== 0:
                continue
            pxy = np.array([len(sy[sy==yval])/float(len(y))  for yval in y_value_list]) #p(x,y)
            t = np.array(pxy[Py>0.]/Py[Py>0.] /Px[i]) # log(P(x,y)/( P(x)*P(y))
            mi += sum(pxy[t>0]*np.log2(t[t>0])) # sum ( P(x,y)* log(P(x,y)/( P(x)*P(y)) )
    except NameError:
        pass
    return mi