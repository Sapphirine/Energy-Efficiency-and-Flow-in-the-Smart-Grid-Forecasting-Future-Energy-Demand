# triplet loss compute
# must define margin in function

def compute_triplet_loss( xpos, xneg, pred):
  margin = 10
  array_len = len(xpos)
  tot = 0
  for i in range(0, array_len):
    
    ppos = pow(xpos[i] - pred[i], 2)
    pneg = pow(xneg[i]- pred[i], 2)
    
    addtotot= max(ppos - pneg + margin, 0)
    
    tot = tot+ max(ppos - pneg + margin, 0)

  return tot
