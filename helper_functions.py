#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[6]:


test_data = pd.DataFrame(np.random.normal(1,1,[100,5]))


# In[7]:


class helper():
    def __init__(self):
        print("initiating")
        
    def trim_data(self,df,limit):
        length = len(df) - limit
        index = np.random.uniform()
        start = int(np.round(index*length,0))
        end = start+limit
        return df.iloc[start:end,:].reset_index(drop=True)
    
    
    def test_classifiers(self,df,x_vars,y_vars,dict_):
        data_train = df.iloc[0:int(np.round(len(df)/2)),:]
        clf=DecisionTreeClassifier(criterion='entropy',max_features=min(len(x_vars),3),class_weight=dict_,min_weight_fraction_leaf=0.01)
        clf=BaggingClassifier(base_estimator=clf,n_estimators=40,max_features=1.,max_samples=1.,oob_score=False)
        fit_items = []
        for item in y_vars:
            print(item)
            X,y = data_train[x_vars],data_train[item]
            print(data_train[item].min())
            #x_test,y_test = data_train[x_vars],data_train[item]
            print("Number of positive y vars is:" + str(len(y[y==1])))
            try:
                fit=clf.fit(X,y)
                x_test = df[x_vars][int(np.round(len(df)/2)):].reset_index(drop=True)
                y_test = df[item][int(np.round(len(df)/2)):].reset_index(drop=True)
                y_hat = fit.predict(x_test)

                results = pd.concat([pd.Series(y_hat), y_test], axis=1, keys=['y_hat', 'y_test'])
                error = len(results)==len(y_test)
                results = results[results['y_test'] == 1]
                #results['y_hat'].replace(-1,0)
                tp_rate = results['y_hat'].sum()/len(results)
                print("Variable is : "+str(item)+".\nScore for test data is " + str(fit.score(X,y)))
                print("the true positive rate is: "+str(tp_rate))
                fit_items.append([fit,fit.score(X,y),tp_rate,len(y[y==1]),len(y_test[y_test==1]),item,error])
            except:
                fit_items.append([None]*5)

        return fit_items

    def test_fitted_classifiers(self,df,x_vars1,x_vars2,x_vars3,x_vars4,y_vars,dict_,models):
        x_vars = x_vars1+x_vars2+x_vars3+x_vars4+['consecutive_vol_increases']
        data_train = df.iloc[0:int(np.round(len(df)/2)),:]
        data_test = df.iloc[int(np.round(len(df)/2)):,:]
        fit_items = []
        for num,model in enumerate(models):
            dic_copy = dict_.copy()
            print(str(num)+"------------------------------------------------------------------")
            filtered_vars = []
            if model is not None:
                imp_=self.featImpMDI(model,featNames=list(x_vars)) #get the MDI
                filtered_vars.append(list(imp_.loc[x_vars1,'mean'].sort_values(ascending=False).index)[0]) #Take the best volume indicator
                filtered_vars.append(list(imp_.loc[x_vars2,'mean'].sort_values(ascending=False).index)[0]) # Take the best price indicator
                filtered_vars.append(list(imp_.loc[x_vars3,'mean'].sort_values(ascending=False).index)[0]) # take the best max increases
                filtered_vars.append(list(imp_.loc[x_vars4,'mean'].sort_values(ascending=False).index)[0]) # take the best max increases
                filtered_vars.append('consecutive_vol_increases')
                rider = True
                while rider == True:
                    clf=DecisionTreeClassifier(criterion='entropy',max_features=min(len(filtered_vars),3),class_weight=dic_copy,min_weight_fraction_leaf=0.01)
                    clf=BaggingClassifier(base_estimator=clf,n_estimators=40,max_features=1.,max_samples=1.,oob_score=False)

                    y_var = y_vars[num]
                    X,y = data_train[filtered_vars],data_train[y_var]
                    print(filtered_vars)

                    print("Number of positive y vars is:" + str(len(y[y==1])))
                    try: # Try to fit the regression
                        fit=clf.fit(X,y)
                        x_test = df[filtered_vars][int(np.round(len(df)/2)):].reset_index(drop=True)
                        y_test = df[y_var][int(np.round(len(df)/2)):].reset_index(drop=True)
                        y_hat = fit.predict(x_test)
                    except: #if you cant fit the regression, then you need to leave the while loop, append, and break the overarching for loop too
                        fit_items.append([None]*8)
                        rider = False
                        break
                        break

                    predict_ratio = np.sum(y_hat)/np.sum(y_test) # update the predict ratio
                    if predict_ratio <1:
                        dic_copy[1] *=2
                    elif predict_ratio >1.2:
                        dic_copy[1] *= 0.8
                    else:
                        rider = False # Break the while loop
                results = pd.concat([pd.Series(y_hat), y_test], axis=1, keys=['y_hat', 'y_test'])
                error = len(results)==len(y_test)
                fp_score = len(results[(results['y_hat'] == 1)&(results['y_test'] == 0)])/results['y_hat'].sum()
                #results['y_hat'].replace(-1,0)
                tp_rate = len(results[(results['y_hat'] == 1)&(results['y_test'] == 1)])/results['y_test'].sum()
                print("Variable is : "+str(y_var)+".\nScore for test data is " + str(fit.score(X,y)))
                print("the true positive rate is: "+str(tp_rate))
                fit_items.append([fit,fit.score(X,y),tp_rate,len(y[y==1]),len(y_test[y_test==1]),y_var,filtered_vars,error,predict_ratio,fp_score])

        return fit_items

    def featImpMDI(self,fit,featNames):
        # feat importance based on IS mean impurity reduction
        df0={i:tree.feature_importances_ for i,tree in         enumerate(fit.estimators_)}
        df0=pd.DataFrame.from_dict(df0,orient='index')
        df0.columns=featNames
        df0=df0.replace(0,np.nan) # because max_features=1
        imp=pd.concat({'mean':df0.mean(),
        'std':df0.std()*df0.shape[0]**-.5},axis=1) # CLT
        imp/=imp['mean'].sum()
        return imp


    def input_float(self,name):
        while True:
            sl = input("Please input the %s (float)\n"% name) 
            try:
                sl = float(sl)
                break
            except:
                print("Cannot convert! Enter float\n")
        return sl

    def input_bool(self,name):
        while True:
            place_order = input("Please input %s (bool)\n" % name)
            if place_order.lower() == "false":
                place_order = False
                break
            elif place_order.lower() == "true":
                place_order = True
                break
            else:
                print("input not accepted\n")
        return place_order

    def input_one_list(self,name):
        while True:
            out_list = input("please input %s as a list separated by commas" % name).split(",")
            try:
                out_list = [float(x) for x in out_list]
                break
            except:
                print("input not accepted")
        return out_list

    def input_two_list(self,name1,name2):
        while True:
            out_list_1 = input("please input %s as a list separated by commas" % (name1)).split(",")
            out_list_2 = input("please input %s as a list separated by commas" % (name2)).split(",")
            if len(out_list_1) == len(out_list_2):
                try:
                    out_list_1 = [float(x) for x in out_list_1]
                    out_list_2 = [float(x) for x in out_list_2]
                    break
                except:
                    print("input not accepted")
            else:
                print("input not accepted")
        return out_list_1,out_list_2

    def input_string(self,name,input_control_usdts,input_control_all):
        while True:
            symbol_ = input("Enter symbol\n")
            symbol_ = str(symbol_)

            if symbol_ in input_control_usdts:
                break
            elif symbol_ in input_control_all:
                print("About to trade non USDT\n")
                break
            else:
                print("Cannot convert! Enter symbol in list!\n")
        return symbol_
    
    def input_any_string(self,name):
        while True:
            output = input("Enter symbol\n")
            return output


    def make_bool(self,df,cols):
        new_df = df.copy(deep = True) ## do not edit in place
        out_list = []
        for item in cols:
            vals = list(new_df[item].dropna().drop_duplicates())

            vals.sort()
            for i,j in enumerate(vals):
                if i==0:
                    print("")
                else:
                    new_df[item+str("_is_")+str(j)+str("_edit")] = 0
                    new_df.at[(new_df[item] ==j) ,item+str("_is_")+str(j)+str("_edit")] = 1 ## Need to create a new column to avoid overwriting
                    out_list.append(item+str("_is_")+str(j)+str("_edit"))
            new_df.drop(labels = (item),axis = 1, inplace = True)
        return new_df,out_list
    
    def is_odd(self,num):
        return num & 0x1
    
    def divide_test_train(self,df,max_lag):
        n,partitions = len(df),int(np.floor(len(df)/(max_lag*2)))
        indices = []
        data_test,data_train  = pd.DataFrame(), pd.DataFrame()
        for i in range(0,partitions):
            indices.append(i*max_lag*2)
        for i,j in enumerate(indices):
            if self.is_odd(i) == 0:
                data_test=data_test.append(df.iloc[i*max_lag*2:i*max_lag*2+max_lag-1,:])
            else:
                data_train=data_train.append(df.iloc[i*max_lag*2:i*max_lag*2+max_lag-1,:])
        data_test.reset_index(drop=True,inplace=True)
        data_train.reset_index(drop=True,inplace=True)
        return data_test,data_train

    
    def generate_combinations(self,list_):
        all_ = list_
        total_combs = 1
        for i in all_:

            total_combs *=len(i) 
        first_ = [[x]*int(total_combs/len(list_[0])) for x in list_[0]]
        first__ = []
        for i in first_:
            first__ = first__+i
        outputs = []
        outputs.append(first__)
        counter = total_combs
        counter_b =  1
        for i,item in enumerate(all_) :
            counter = int(counter /len(item))
            counter_b*= len(item)
            if i >0:

                rider = [[x]*counter for x in item]
                rider2 = []
                for j in rider:
                    rider2 = rider2+j
                rider2 = rider2*int(counter_b/len(item))
                outputs.append(rider2)
        return pd.DataFrame(outputs).transpose()
    
    def calibrate_function(self,list_,function):
        combinations = self.generate_combinations(list_)
        output_values,inputs = [],[]
        for i in range(0,len(combinations)):
            arguments = list(combinations.iloc[i,:])
            output = function(len(list_),arguments)
            output_values.append(output)
        return [combinations,output_values]


# In[8]:


#print("Testing helper")

