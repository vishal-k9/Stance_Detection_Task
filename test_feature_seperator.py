#seperates the features in the datasets 
import csv
import sys


f_obj=open("testdata.txt","rb")
f2=csv.reader(f_obj)

header=next(f2)
# next(header)
file=header[0].split('\t')


# print file	

# for row in f:
# 	print row
# 	break	
c=0

with open("testdata.txt","rb") as f,open("test dataset/"+file[0]+".csv","wb") as f1,open("test dataset/"+file[1]+".csv","wb") as f2,\
open("test dataset/"+file[2]+".csv","wb") as f3,\
open("test dataset/"+file[3]+".csv","wb") as f4,open("test dataset/"+file[1]+"_"+file[3]+".csv","wb") as f5, open("test dataset/"+file[2]+"_"+ file[1]+ ".csv","wb") as f6, open("test dataset/"+file[2]+"_without_hashtags"+".csv","wb") as f7:
	next(f)
	for row in f:
		
		row=row.split('\t')


		i=0
		temp=""

		for col in row:
			
			if(i==0):
				f1.write(col+"\n")
			elif(i==1):
				temp=col
				f2.write(col+"\n")
				f5.write(col+"\t")
			elif(i==2):
				# if(c==0):
				# 	print col.split()
				# 	c=1
				l=col.split()
				l2=[]
				for word in l:
					if(word[0]!='#'):
						l2.append(word)
				tweet=' '.join(word for word in l2)
				f7.write(tweet+'\n')		
				f3.write(col+"\n")
				f6.write(col+" "+temp+"\n")
			elif(i==3):
				f4.write(col)
				f5.write(col)		
			i+=1	

f_obj.close()