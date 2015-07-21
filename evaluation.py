import numpy as np
import my_accumarray as accum

def compute_overlap_sprep(sp, reg2sp, inst):
  #compute intersection between all instances and all sp
  intsp = accum.my_accumarray((inst.reshape(-1), sp.reshape(-1)-1),1, (np.max(inst)+1,np.max(sp)))
  
  #compute the total intersection
  R=reg2sp.astype(np.float32, copy=True)
  inters = np.dot(intsp, R)
  
  #to compute the union, compute region areas
  spareas = accum.my_accumarray(sp.reshape(-1)-1,1,np.max(sp))
  totalareas = np.dot(spareas.reshape(1,-1),R)

  #compute instance areas
  instareas = accum.my_accumarray(inst.reshape(-1),1,np.max(inst)+1)
  #union is sum of areas - inters
  uni = instareas.reshape(-1,1)+totalareas.reshape(1,-1)-inters
  #union that is 0 should be set to 1 for safety
  uni[uni==0]=1.
  ov = inters/uni
  return ov

def assign_dets_to_gt(scr, ov, thresh):
  #assign everything
  assigned = np.argmax(ov,0)
  bestov = np.max(ov,0)
  #now go down ranked list
  covered = [False]*ov.shape[0]
  labels = np.zeros(ov.shape[1])
  idx = np.argsort(-scr)
  for i in idx:
    if not covered[assigned[i]] and bestov[i]>=thresh:
      covered[assigned[i]]=True
      labels[i] = 1
  return labels


def generalized_det_eval_simple(scores, ov, gt, categid, thresh):
  #we will count and gt first
  numdets = 0
  numgt = 0  
  for k in range(len(scores)):
    categids = gt[k]['classes']
    numdets += scores[k].size
    numgt += np.sum(categids==categid)

  #init
  allscores = np.zeros(numdets)
  alllabels = np.zeros(numdets)
  count = 0
  for k in range(len(scores)):
    #if no dets here continue
    if scores[k].size==0:
      continue
    #otherwise, assign
    labelstmp = assign_dets_to_gt(scores[k],ov[k][gt[k]['classes']==categid,:], thresh)
    allscores[count:count+scores[k].size] = scores[k]
    alllabels[count:count+scores[k].size] = labelstmp
    count +=scores[k].size

  #compute PR
  [ap, prec, rec] = getPR(allscores, alllabels, numgt)
  
  return ap, prec, rec



