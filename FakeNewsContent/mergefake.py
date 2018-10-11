
# coding: utf-8

# In[1]:


import json
import pandas as pd
import glob
import numpy as np


# In[11]:


result2 = []
indx = []
for f2 in glob.glob("*.json"):
    if f2=="fakemerged_file.json": 
        continue  
    with open(f2, "r") as infile2:
#         result2.append(f2)
        result2.append(json.load(infile2))
        string1, string2, string3 = f2.split('_')
        string4, string5 = string3.split('-')
        indx.append(string4)
        
with open("fakemerged_file.json", "w") as outfile2:
     json.dump(result2, outfile2)
        


# In[12]:


d =pd.read_json('fakemerged_file.json')
df = pd.DataFrame(d)


# In[13]:


df = df.assign(Post=indx)


# In[14]:


df


# In[15]:


# Add missed dates
df['publish_date'][20]={'$date': 1513947565000}
df['publish_date'][5]={'$date': 1472860800000}
df['publish_date'][8]={'$date': 1488672000000}
df['publish_date'][9]={'$date': 1491091200000}
df['publish_date'][19]={'$date': 1488153600000}
df['publish_date'][23]={'$date': 1492992000000}
df['publish_date'][25]={'$date': 1493856000000}
df['publish_date'][26]={'$date': 1482451200000}
df['publish_date'][28]={'$date': 1490313600000}
df['publish_date'][91]={'$date': 1491091200000}
df['publish_date'][98]={'$date': 1497484800000}
df['publish_date'][99]={'$date': 1488092400000}
df['publish_date'][100]={'$date': 1483340400000}
df['publish_date'][102]={'$date': 1493362800000}
df['publish_date'][105]={'$date': 1463356800000}
df['publish_date'][108]={'$date': 1493164800000}
df['publish_date'][109]={'$date': 1482192000000}
df['publish_date'][118]={'$date': 1490313600000}
df['publish_date'][31]={'$date': 1499472000000}
df['publish_date'][35]={'$date': 1499040000000} # prosoxi
df['publish_date'][39]={'$date': 1496361600000}
df['publish_date'][42]={'$date': 1486512000000}
df['publish_date'][44]={'$date': 1492387200000}
df['publish_date'][46]={'$date': 1480982400000}
df['publish_date'][47]={'$date': 1490054400000}
df['publish_date'][53]={'$date': 1486080000000}
df['publish_date'][56]={'$date': 1449532800000}
df['publish_date'][61]={'$date': 1491436800000}
df['publish_date'][63]={'$date': 1495497600000}
df['publish_date'][68]={'$date': 1479340800000}
df['publish_date'][75]={'$date': 1485820800000}
df['publish_date'][76]={'$date': 1483920000000}
df['publish_date'][80]={'$date': 1486339200000}
df['publish_date'][81]={'$date': 1487203200000}
df['publish_date'][82]={'$date': 1485388800000}
df['publish_date'][88]={'$date': 1486771200000}
df['publish_date'][89]={'$date': 1494892800000}
df['publish_date'][32]={'$date': 1498176000000}

df.loc[df['publish_date'].isnull()]


# In[231]:


# df['publish_date']


# In[16]:


for index, row in df.iterrows():
#     print(list(df['publish_date'][index].values())[0])
    df['publish_date'][index]=list(df['publish_date'][index].values())[0]
    


# In[17]:


df = df.sort_values('publish_date', axis=0)
sortedposts = df['Post']


# In[18]:


sortedposts = sortedposts.reset_index(drop=True)
sortedposts = sortedposts.astype(int)


# In[19]:


np.savetxt('../sortedfakeposts.txt', sortedposts)

