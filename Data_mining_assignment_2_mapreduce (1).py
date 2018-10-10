
# coding: utf-8

# #MAPPER1

# In[ ]:


import sys
f = open("mapper_one_output.txt","w+")      
for line in sys.stdin:
    #Get only one line of the input data at a time
    #Remove leading and trailing whitespace---
    line = line.strip()
    try:
        #Split the input line
        words = line.split(',')[4]
        f.write('%s \n' % (words))
    except IndexError:
        continue
f.close()


# #MAPPER2

# In[ ]:


import sys
f = open("mapper_two_output.txt","w+")      
for line in sys.stdin:
    #Get only one line of the input data at a time
    #Remove leading and trailing whitespace---
    line = line.strip()
    try:
        #Split the input line
        word1 = line.split(',')[0]
        word2 = line.split(',')[4]
        words = str(word1)+','+str(word2)
        f.write('%s \n' % (words))
    except IndexError:
        continue
f.close()


# #MAPPER3

# In[ ]:


import sys
f = open("mapper_three_output.txt","w+")      
for line in sys.stdin:
    #Get only one line of the input data at a time
    #Remove leading and trailing whitespace---
    line = line.strip()
    try:
        #Split the input line
        word1 = line.split(',')[6]
        word2 = line.split(',')[4]
        words = str(word1)+','+str(word2)
        f.write('%s \n' % (words))
    except IndexError:
        continue
f.close()


# #MAPPER4

# In[ ]:


import sys
f = open("mapper_four_output.txt","w+")      
for line in sys.stdin:
    #Get only one line of the input data at a time
    #Remove leading and trailing whitespace---
    line = line.strip()
    try:
        #Split the input line
        word1 = line.split(',')[0]
        word2 = line.split(',')[6]
        word3 = line.split(',')[4]
        words = str(word1)+','+str(word2)+','+str(word3)
        f.write('%s \n' % (words))
    except IndexError:
        continue
f.close()


# #REDUCER

# In[ ]:


import sys
f = open("reducer_one_output.txt","w+")  
consumption_freq ={}

for line in sys.stdin:
    line=line.strip()
    try:
        consumption_freq[line] +=1
    except:
        consumption_freq[line] =1

for consumption in consumption_freq.keys():
    f.write("%s,%s \n"%(consumption, consumption_freq[consumption]))


# In[ ]:


#CLEANING


# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import plotly.plotly as py
import plotly.tools as tls
import plotly.graph_objs as go
import plotly
plotly.offline.init_notebook_mode(connected=True)
import plotly.offline as py


# In[30]:


reducer_one_output


# In[151]:


reducer_one_output = pd.read_csv("C:/Users/Acer/Downloads/map_reduce/reducer_one_output.txt") 
reducer_two_output = pd.read_csv("C:/Users/Acer/Downloads/map_reduce/reducer_two_output.txt") 
reducer_three_output = pd.read_csv("C:/Users/Acer/Downloads/map_reduce/reducer_three_output.txt") 
reducer_four_output = pd.read_csv("C:/Users/Acer/Downloads/map_reduce/reducer_four_output.txt") 
reducer_one_output  
reducer_one_output.columns= ['Consumption','Frequency']
reducer_one_output_no_missing = reducer_one_output.dropna()
reducer_one_output_no_missing
reducer_one_output_no_missing['Consumption'] = pd.to_numeric(reducer_one_output_no_missing['Consumption'] , errors='coerce')
reducer_one_output_no_missing['Frequency'] = pd.to_numeric(reducer_one_output_no_missing['Frequency'] , errors='coerce')
reducer_one_output_no_missing_no_negative =reducer_one_output_no_missing[(reducer_one_output_no_missing['Consumption']>0) & (reducer_one_output_no_missing['Frequency']>0 ) ]
reducer_one_output_no_missing_no_negative


# In[118]:


reducer_two_output


# In[117]:


#reducer_two_output.columns= ['Connection Type','Consumption','Frequency']
#reducer_two_output_no_missing = reducer_two_output.dropna()
#reducer_two_output_no_missing
#reducer_two_output_no_missing['Consumption'] = pd.to_numeric(reducer_two_output_no_missing['Consumption'] , errors='coerce')
#reducer_two_output_no_missing['Frequency'] = pd.to_numeric(reducer_two_output_no_missing['Frequency'] , errors='coerce')
reducer_two_output_no_missing_no_negative=reducer_two_output_no_missing[(reducer_two_output_no_missing> 0).all(1)]
reducer_two_output_no_missing_no_negative


# In[10]:


reducer_three_output


# In[135]:


reducer_three_output.columns= ['Bore Well','Consumption','Frequency']
reducer_three_output_no_missing = reducer_three_output.dropna()
reducer_three_output_no_missing
reducer_three_output_no_missing['Consumption'] = pd.to_numeric(reducer_three_output_no_missing['Consumption'] , errors='coerce')
reducer_three_output_no_missing['Frequency'] = pd.to_numeric(reducer_three_output_no_missing['Frequency'] , errors='coerce')
reducer_three_output_no_missing_no_negative=reducer_three_output_no_missing[(reducer_three_output_no_missing> 0).all(1)]
reducer_three_output_no_missing_no_negative
reducer_three_output_no_missing_no_negative_boolean=reducer_three_output_no_missing_no_negative[(reducer_three_output_no_missing_no_negative['Bore Well'] == "TRUE") | (reducer_three_output_no_missing_no_negative['Bore Well'] == "FALSE")]
reducer_three_output_no_missing_no_negative_boolean


# In[11]:


reducer_four_output


# In[136]:


reducer_four_output.columns= ['Connection Type','Bore Well','Consumption','Frequency']
reducer_four_output_no_missing = reducer_four_output.dropna()
reducer_four_output_no_missing
reducer_four_output_no_missing['Consumption'] = pd.to_numeric(reducer_four_output_no_missing['Consumption'] , errors='coerce')
reducer_four_output_no_missing['Frequency'] = pd.to_numeric(reducer_four_output_no_missing['Frequency'] , errors='coerce')
reducer_four_output_no_missing_no_negative=reducer_four_output_no_missing[(reducer_four_output_no_missing> 0).all(1)]
reducer_four_output_no_missing_no_negative
reducer_four_output_no_missing_no_negative_boolean=reducer_four_output_no_missing_no_negative[(reducer_four_output_no_missing_no_negative['Bore Well'] == "TRUE") | (reducer_four_output_no_missing_no_negative['Bore Well'] == "FALSE")]
reducer_four_output_no_missing_no_negative_boolean


# In[141]:


reducer_one_output_no_missing_no_negative.dtypes


# In[142]:


reducer_two_output_no_missing_no_negative.dtypes


# In[143]:


reducer_three_output_no_missing_no_negative_boolean.dtypes


# In[145]:


reducer_four_output_no_missing_no_negative_boolean.dtypes


# #PLOTTING

# In[164]:


sns.distplot(reducer_one_output_no_missing_no_negative['Consumption'], hist=False, rug=True);
sns.distplot(reducer_one_output_no_missing_no_negative['Frequency'], hist=False, rug=True);


# In[172]:


import numpy as np
import matplotlib.pyplot as plt

def make_hist(ax, x, bins=None, binlabels=None, width=0.85, extra_x=1, extra_y=4, 
              text_offset=0.3, title=r"Frequency diagram", 
              xlabel="Values", ylabel="Frequency"):
    if bins is None:
        xmax = max(x)+extra_x
        bins = range(xmax+1)
    if binlabels is None:
        if np.issubdtype(np.asarray(x).dtype, np.integer):
            binlabels = [str(bins[i]) if bins[i+1]-bins[i] == 1 else 
                         '{}-{}'.format(bins[i], bins[i+1]-1)
                         for i in range(len(bins)-1)]
        else:
            binlabels = [str(bins[i]) if bins[i+1]-bins[i] == 1 else 
                         '{}-{}'.format(*bins[i:i+2])
                         for i in range(len(bins)-1)]
        if bins[-1] == np.inf:
            binlabels[-1] = '{}+'.format(bins[-2])
    n, bins = np.histogram(x, bins=bins)
    patches = ax.bar(range(len(n)), n, align='center', width=width)
    ymax = max(n)+extra_y

    ax.set_xticks(range(len(binlabels)))
    ax.set_xticklabels(binlabels)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ymax)
    ax.grid(True, axis='y')
    # http://stackoverflow.com/a/28720127/190597 (peeol)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # http://stackoverflow.com/a/11417222/190597 (gcalmettes)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    autolabel(patches, text_offset)

def autolabel(rects, shift=0.3):
    """
    http://matplotlib.org/1.2.1/examples/pylab_examples/barchart_demo.html
    """
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            plt.text(rect.get_x()+rect.get_width()/2., height+shift, '%d'%int(height),
                     ha='center', va='bottom')


# In[222]:


fig, ax = plt.subplots(figsize=(14,5))
make_hist(ax, reducer_one_output_no_missing_no_negative['Consumption'], bins=list(range(10))+list(range(10,41,5))+[np.inf], extra_y=6)
plt.show()


# In[219]:


plt.plot(reducer_one_output_no_missing_no_negative['Consumption'], reducer_one_output_no_missing_no_negative['Frequency'],'o')


# In[214]:


sns.lmplot('Consumption', 'Frequency', data=reducer_two_output_no_missing_no_negative, hue='Connection Type', fit_reg=False)


# In[228]:


sns.lmplot('Frequency', 'Consumption', data=reducer_two_output_no_missing_no_negative, hue='Connection Type', fit_reg=True)


# In[227]:



sns.lmplot('Frequency', 'Consumption', data=reducer_four_output_no_missing_no_negative_boolean, hue='Bore Well', fit_reg=True)


# In[229]:


def plotter(index,reducer_four_output_no_missing_no_negative_boolean,i):
    df.columns = ['Frequency']
    df.index = index
    df.plot.bar(label="%s"%(i),ls='dashed',alpha=0.5)
    plt.title("The Histogram for Connection_Type and Borewell %s"%(i))
    plt.legend()
# Set bar height dependent on country extension
# Set min and max bar thickness (from 0 to 1)
hmin, hmax = 0.3, 0.9
xmin, xmax = min(df['Consumption']), max(df['Consumption'])
# Function that interpolates linearly between hmin and hmax
f = lambda x: hmin + (hmax-hmin)*(x-xmin)/(xmax-xmin)
# Make array of heights
hs = [f(x) for x in df['Consumption']]
 
# Iterate over bars
for container in ax.containers:
    # Each bar has a Rectangle element as child
    for i,child in enumerate(container.get_children()):
        # Reset the lower left point of each bar so that bar is centered
        child.set_y(child.get_y()- 0.125 + 0.5-hs[i]/2)
        # Attribute height to each Recatangle according to country's size
        plt.setp(child, height=hs[i])
df['Connection Type'] = df[0].apply(str)+" "+df[1].apply(str)
del df[0]
del df[1]
    
for i in reducer_four_output_no_missing_no_negative_boolean['Connection Type'].unique():
    plotter(reducer_four_output_no_missing_no_negative_boolean[reducer_four_output_no_missing_no_negative_boolean['Connection Type']==i][:][3].values, df[df['Connection Type']==i][:][3],i)
plt.show()

plotter(reducer_four_output_no_missing_no_negative_boolean[reducer_four_output_no_missing_no_negative_boolean['Connection Type']=="All Kinds of Hotels False"][:][3].values, reducer_four_output_no_missing_no_negative_boolean[reducer_four_output_no_missing_no_negative_boolean['Connection Type']=="All Kinds of Hotels False"][:][3],"All Kinds of Hotels False")
plt.show()

plt.clf()

