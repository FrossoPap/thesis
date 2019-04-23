
# coding: utf-8

# In[1]:


import json
import pandas as pd
import glob
import numpy as np


# In[2]:


result2 = []
indx = []
for f2 in glob.glob("*.json"):
    if f2=="realmerged_file.json": 
        continue  
    with open(f2, "r") as infile2:
#         result2.append(f2)
        result2.append(json.load(infile2))
        string1, string2, string3 = f2.split('_')
        string4, string5 = string3.split('-')
        indx.append(string4)
        
with open("realmerged_file.json", "w") as outfile2:
     json.dump(result2, outfile2)
        


# In[3]:


d =pd.read_json('realmerged_file.json')
df = pd.DataFrame(d)


# In[4]:


df = df.assign(Post=indx)


# In[5]:


df


# In[6]:


print(df.keys())


# In[7]:


# # Add missed dates
df['publish_date'][4]={'$date': 1473469261000}
df['publish_date'][10]={'$date': 1474506061000}
df['publish_date'][11]={'$date': 1474938061000}
df['publish_date'][14]={'$date': 1483228800000} # akiro news
df['publish_date'][17]={'$date': 1474253760000}
df['publish_date'][19]={'$date': 1474253760000}
df['publish_date'][20]={'$date': 1503975360000} #prosoxi check
df['publish_date'][21]={'$date': 1469415360000}
df['publish_date'][24]={'$date': 1474858560000} 
df['publish_date'][29]={'$date': 1474972010000}
df['publish_date'][30]={'$date': 1503881640000}
df['publish_date'][34]={'$date': 1474494840000}
df['publish_date'][36]={'$date': 1474577640000}
df['publish_date'][44]={'$date': 1474247880000}
df['publish_date'][46]={'$date': 1474381080000}
df['publish_date'][47]={'$date': 1474568460000}
df['publish_date'][51]={'$date': 1474329600000}
df['publish_date'][54]={'$date': 1474243200000}
df['publish_date'][65]={'$date': 1474934400000}
df['publish_date'][67]={'$date': 1474416000000} 
df['publish_date'][69]={'$date': 1503619200000}
df['publish_date'][71]={'$date': 1474934400000}
df['publish_date'][72]={'$date': 1474411080000}
df['publish_date'][80]={'$date': 1474398480000}
df['publish_date'][81]={'$date': 1474645080000} # not sure, check
df['publish_date'][86]={'$date': 1474904280000}
df['publish_date'][87]={'$date': 1474569900000}
df['publish_date'][94]={'$date': 1474589700000}
df['publish_date'][96]={'$date': 1472688900000} # post akiro
df['publish_date'][97]={'$date': 1474366500000}
df['publish_date'][98]={'$date': 1475020800000}
df['publish_date'][103]={'$date': 1474629210000}
df['publish_date'][107]={'$date': 1474888410000}
df['publish_date'][108]={'$date': 1470012842000}
df['publish_date'][111]={'$date': 1474548842000}
df['publish_date'][112]={'$date': 1474329600000}
df['publish_date'][113]={'$date': 1474502400000}
df['publish_date'][115]={'$date': 1474243200000}
df['publish_date'][117]={'$date': 1466348210000}

df.loc[df['publish_date'].isnull()]


# In[8]:


# df['publish_date']


# In[9]:


for index, row in df.iterrows():
#     print(list(df['publish_date'][index].values())[0])
    df['publish_date'][index]=list(df['publish_date'][index].values())[0]
    


# In[10]:


df = df.sort_values('publish_date', axis=0)
sortedposts = df['Post']


# In[12]:


sortedposts = sortedposts.reset_index(drop=True)
sortedposts = sortedposts.astype(int)


# In[13]:


np.savetxt('../sortedrealposts.txt', sortedposts)

