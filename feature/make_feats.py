import os
import sys
#fw=open('label_2077_5dim.txt','w')
#fp_label=open('label_2077_5dim_20200929.scp', 'r')

fp_label=open(sys.argv[1],'r') 
fw=open(sys.argv[2],'w')


lists=[]
for i in os.listdir(sys.argv[3]):
    lists.append(i)

count=0
for line in fp_label.readlines():
    filename = line.split()[0].replace('.wav','')
    mos = ' '.join(line.split()[1:])
    if filename+'.npy' in lists:
        fw.write(filename+' '+sys.argv[3]+'/'+filename+'.npy '+mos+'\n')
        count+=1
    else:
        pass
        #print(line)
#print(count)
