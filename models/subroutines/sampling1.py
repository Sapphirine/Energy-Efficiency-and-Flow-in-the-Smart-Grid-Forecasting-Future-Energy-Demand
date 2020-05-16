# orig sampling method from unsupervised scalable rep. learning paper
def sampling1(y, N, K): # sequence y_i for i in 1 to N
  # size of y
  for i in range(1,N):
    yi = y[i]
    ysize = len(yi)
    # pick sizes s_pos and s_ref - unif at random 
    spos = np.random.randint(1, ysize)
    sref = np.random.randint(spos, ysize) 
    # pick xref, xpos 
    # pick starting index among 0 to (ysize - sref)
    startref = np.random.randint(0, ysize - sref)
    startpos = np.random.randint(0, ysize - spos)
    xref = yi[startref:startref+sref]
    xpos = yi[startpos:startpos+spos]
    xneg= []
    for k in range(1, K): 
      ik = np.random.randint(1, N)
      sneg = np.random.randint(1,len(y[ik]))
      # pick xneg
      startneg = np.random.randint(0, ysize-sneg)
      yk = y[k]
      xneg_k = yk[startneg:sneg]
      xneg.append(xneg_k)
    return( xref, xpos, xneg)
