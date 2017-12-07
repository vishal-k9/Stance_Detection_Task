from sklearn.metrics import f1_score

y_true=[]
y_pred=[]

with open("Stance.csv","rb") as f1, open("pred.csv","rb") as f2:
	for row in f1:
		y_true.append(row.strip())
	for row in f2:
		y_pred.append(row.strip())

# print y_true[:10]
# print y_pred[:10]			

f=f1_score(y_true,y_pred,average=None)
f1=(f[0]+f[1])/2
print f1
