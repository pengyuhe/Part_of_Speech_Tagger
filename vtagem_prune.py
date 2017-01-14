
import math
import sys,os
import numpy as np
from numpy import logaddexp
from copy import deepcopy,copy
from heapq import nlargest, heappush, heappop
import heapq
class Tagger:

    def __init__(self):
        
        ### The possibilities
        self.P_Change    = {}
        self.P_Emission = {}

        ### The posterior possibilities that we will make use of to estimate the new possibilities

        self.P_new_Tag={}
        self.P_new_Transfer={}

        self.Ctt_Train={}
        self.Cwt_Train={}
        
        self.OCS_iter=0

        
        self.Ctt = {}
        self.Cwt = {}
        ### Count of words in Training, only used in one-count smoothing
        self.Cword_Train ={}
        self.Cword={}
        self.Tags = set([])
        self.Vocab = set([]) ### All the words in training and raw data
                            ### for test data, we just need to add one 'oov' in that
                            ### Equal to the union of Raw and Train words + 'oov' 

        self.Known_dict= set([])
        self.Seen_dict = set([])
        self.Novel_dict= set([])

        self.Train_data = []
        self.Train_tag = []

        ### Test_data will use to run decoders 
        ### and regard words out of Known_dict as oov
        ### But Test_original will keep the original lexicons 
        ### to judge is his known, seen or novel
        self.Test_data = []
        self.Test_tag  = []
        self.Test_original_data=[]

        
        self.Raw_data = []
        self.Raw_tag  = []
        
        ### The tag dictionary, the key are the lexicons, the value is a list of all possible tags for this key (Tag_Dict[oov word] = [all tags] 
        self.Tag_Dict={}

        self.sing_tt={}
        self.sing_wt={}

        self.Result_Tag=[]

    def AddCword(self, key):
        if key not in self.Cword:
            self.Cword[key]=1.0
        else:
            self.Cword[key]+=1.0

    def AddCtt(self, key):
        if key not in self.Ctt:
            self.Ctt[key]=1.0
        else:
            self.Ctt[key]+=1.0
            
    def AddCwt(self, key):
        if key not in self.Cwt:
            self.Cwt[key]=1.0
        else:
            self.Cwt[key]+=1.0

    def Read_Train(self,Train_file):

        with open(Train_file,'r') as f:
            Train_Content=f.readlines()

        LC=len(Train_Content)

        for i in range(0,LC):

            newline  = Train_Content[i]
            newline  = newline.strip('\n')
            newSline = newline.split('/')

            new_data = newSline[0]
            new_tag  = newSline[1]
            
            self.Train_data.append(new_data)
            self.Train_tag.append(new_tag)
            
            ### add the new data in the data (word)  set
            ### for the statistics of seen and unseen words

            if new_data not in self.Tag_Dict:
                self.Tag_Dict[new_data]=[new_tag]
            else:
                if new_tag not in self.Tag_Dict[new_data]:
                    self.Tag_Dict[new_data].append(new_tag)

            ### Add the possible new data into the "Data" set
            if new_data not in self.Vocab:
                self.Vocab.add(new_data)

            ### add the new tag to the training tag set
            if new_tag not in self.Tags:
                self.Tags.add(new_tag)
            

            ### No matter what, add the emission count

            self.Cwt_Train[(new_data,new_tag)]=self.Cwt_Train.get( (new_data,new_tag),0)+1.0
            self.Cwt_Train[new_tag]=self.Cwt_Train.get( new_tag,0)+1.0

            

            self.AddCword(new_data)
            # Change sing_wt
            if (self.Cwt_Train[(new_data,new_tag)]==1.0):
                if new_tag in self.sing_wt:
                    self.sing_wt[new_tag]+=1.0
                else:
                    self.sing_wt[new_tag]=1.0
            elif(self.Cwt_Train[(new_data,new_tag)]==2.0):
                self.sing_wt[new_tag]-=1.0

            ### other than the first case, add the Change count
            if(i==0):
                old_tag  = new_tag
                continue

            else:

                self.Ctt_Train[(new_tag,old_tag)]=self.Ctt_Train.get( (new_tag,old_tag),0)+1.0
                self.Ctt_Train[old_tag]=self.Ctt_Train.get( old_tag,0)+1.0



                # Change sing_tt
                if (self.Ctt_Train[(new_tag,old_tag)]==1.0):
                    if old_tag in self.sing_tt:
                        self.sing_tt[old_tag]+=1.0
                    else:
                        self.sing_tt[old_tag]=1.0
                elif(self.Ctt_Train[(new_tag,old_tag)]==2.0):
                    self.sing_tt[old_tag]-=1.0



                old_tag=new_tag

        self.Known_dict=deepcopy(self.Vocab)

        self.Ctt=deepcopy(self.Ctt_Train)
        self.Cwt=deepcopy(self.Cwt_Train)

        self.Cword_Train=deepcopy(self.Cword)

    def Read_Raw(self,Raw_file):

        with open(Raw_file,'r') as f:
            Raw_Content=f.readlines()
        
        LC=len(Raw_Content)
    
        for i in range(0,LC):

            newline  = Raw_Content[i]
            newline  = newline.strip('\n')

            new_data = newline

            if(new_data not in self.Known_dict):
                if(new_data not in self.Seen_dict):
                    self.Seen_dict.add(new_data)

            ### If we add raw to the lexicon count
            self.AddCword(new_data)
                

            self.Raw_data.append(new_data)

            if(new_data not in self.Vocab): ### if we found a novel word, allow all the tags.
                self.Tag_Dict[new_data]=[w for w in self.Tags if w != '###'] 
                self.Vocab.add(new_data)


    def Read_Test_oov(self,Test_file):

        with open(Test_file,'r') as f:
            Test_Content=f.readlines()
        
        LC=len(Test_Content)
    
        for i in range(0,LC):

            newline  = Test_Content[i]
            newline  = newline.strip('\n')
            newSline = newline.split('/')

            new_data = newSline[0]
            new_tag  = newSline[1]


            self.Test_original_data.append(new_data)

            if(new_data not in self.Vocab):
                
                self.Novel_dict.add(new_data)

                new_data='oov'
                self.Vocab.add('oov')
                self.Tag_Dict[new_data]=[w for w in self.Tags if w != '###'] 

            ### These are two lists that simply restore the data by index
            self.Test_data.append(new_data)
            self.Test_tag.append(new_tag)



    def One_count_Smooth(self): ### Note: if we want to use this smoothing
                                ### We must use Read_Test_oov function in __init__
        
        ### different kinds of data in both train and raw, plus an 'oov'
        ### We need temporarily substract '###' because we skipped it
        V_data=len(self.Vocab)-1.0

        
        if(self.OCS_iter!=0):
            N_tt=len(self.Train_data)+len(self.Raw_data)-2.0
            Word_Count=self.Cword
            N_wt=len(self.Train_data)+len(self.Raw_data)-self.Cword['###']
        else:
            N_tt=len(self.Train_data)-1.0
            Word_Count=self.Cword_Train
            N_wt=len(self.Train_data)-self.Cword_Train['###'] 

        p_wt_backoff={}
        p_tt_backoff={}
        for tag in self.Tags:
            if(tag=='###'):
                continue
            Lambda=1.0+self.sing_wt[tag]
            
            P_temp_tag=0.0
            V_temp=0.0
            Cword_temp=0.0
            for data in self.Vocab:
                if(data=='###'):
                    continue
                if (data,tag) in self.Cwt:### In the first loop Cwt=Cwt_Train
                                          ### No need to worry
                    Temp_num=self.Cwt[(data,tag)]
                else:
                    Temp_num=0.0 ### in here we smoothed those "seen" words
                               ### that only appeared 


                if(data in Word_Count):
                    p_wt_backoff[(data,tag)] = (Word_Count[data]+1.0)/(N_wt + V_data)
                    Cword_temp+=Word_Count[data]
                else:
                    p_wt_backoff[(data,tag)] = 1.0 / (N_wt + V_data)
                    
                V_temp+=1.0
                P_temp_tag+=p_wt_backoff[(data,tag)]


                Lambda_P=Lambda * p_wt_backoff[(data,tag)]

                self.P_Emission[ (data,tag) ]=( Temp_num + Lambda_P)/( self.Cwt[tag]+ Lambda)
                '''
                ### Test Case
                if(data=='1' and tag=='C'):
                    print('Lambda=',Lambda)
                    print('Cwt[C]=',self.Cwt['C'])
                    print('Cwt[(1,C)]=',self.Cwt[('1','C')])
                    print('p_wt_backoff[(data,tag)]=',p_wt_backoff[(data,tag)])
                    print("V_data=",V_data)
                    print("N=",N_wt)
                '''
                            
            
            #print("Cword_temp=",Cword_temp,'N',N)
            #print("Vtemp",V_temp,'V_data',V_data)
            #print(tag,P_temp_tag)

        self.P_Emission[('###','###')]=1.0
        
        for old_tag in self.Tags:
            Lambda=1+self.sing_tt[old_tag]
            
            Temp_P_tt=0.0
            for new_tag in self.Tags:
                if (new_tag,old_tag) in self.Ctt:
                    Temp_num=self.Ctt[ (new_tag,old_tag)]
                else:
                    Temp_num=0.0
                
                p_tt_backoff[(new_tag,old_tag)]=self.Ctt[new_tag]/N_tt
                Temp_P_tt+=p_tt_backoff[(new_tag,old_tag)]
            
                Lambda_p=Lambda*p_tt_backoff[(new_tag,old_tag)]

                self.P_Change[(new_tag,old_tag)]=(Temp_num +Lambda_p)/( self.Ctt[old_tag]+Lambda)   

            #print("old_tag=",old_tag,Temp_P_tt)


        #self.Check_Emission()
        #self.Check_Change()

    def Add_lambda_Smooth(self,Lambda):
        ### V is the num of different words except '###'
        V_data=len(self.Vocab)-1.0
        for tag in self.Tags:
            if(tag=='###'):
                continue
            for data in self.Vocab:
                if(data=='###'):
                    continue

                if (data,tag) in self.Cwt:
                    Temp_num=self.Cwt[(data,tag)]
                else:
                    Temp_num=0.0

                self.P_Emission[ (data,tag) ]=( Temp_num + Lambda)/( self.Cwt[tag]+V_data*Lambda)

        self.P_Emission[('###','###')]=1.0

        V_tag=len(self.Tags)
        for new_tag in self.Tags:
            for old_tag in self.Tags:
                if (new_tag,old_tag) in self.Ctt:
                    Temp_num=self.Ctt[ (new_tag,old_tag)]
                else:
                    Temp_num=0.0
                    

                self.P_Change[(new_tag,old_tag)]=(Temp_num +Lambda)/( self.Ctt[old_tag]+V_tag*Lambda)   

        '''
        selfCheck_Emission()
        self.Check_Change()
        '''     

    ### We will base on the Pruning_rate and the Prob (i)  to Prune the Current_Tag_Dict 
    def Prune(self,Pruning_rate,Prob,Current_Tag_Dict,i):
        
        Temp_Tag_Dict=[]
        if(Pruning_rate == 1):
            return Temp_Tag_Dict

        elif( len( Current_Tag_Dict ) <=3 ):
            return Temp_Tag_Dict

        else:
            Temp_p_ti=[]
            for ti in Current_Tag_Dict:
                if(Prob[(ti,i)] != 'e'):
                    heappush( Temp_p_ti , [-Prob[(ti,i)],ti] )

            Pruning_N = int( math.ceil( Pruning_rate*len(Current_Tag_Dict) ))

            for Pop_N in range(0,Pruning_N):
                Temp_node=heappop(Temp_p_ti)
                Temp_Tag_Dict.append(Temp_node[1])

            return Temp_Tag_Dict
    def Viterbi(self,Pruning_rate):

        ### L_test contains start and end ###
        L_test=len(self.Test_data)
        L_Tags=len(self.Tags)

        Temp_Tag_Dict=[]
        mu={}
        backpointer={}

        for ti in self.Tags:
            for i in range(0,L_test):
                mu[ (ti,i)]='e' ### this means 'minus infinity'

        mu[('###',0)]= math.log(1) # log version

        #mu[('###',0)]= 1  # probability version #i## Test_data[0]=Test_tag[0]='###'

        for i in range(1,L_test):
            
            #if(i%1000 ==0):
            #   print("%d / %d " %(i,L_test))
            '''
            print("old_mu is") 
            for ti_1 in self.Tag_Dict[self.Test_data[i-1]]:
                print(mu[(ti_1,i-1)])
            '''
            if (not Temp_Tag_Dict):
                Local_Tag_Dict=deepcopy(self.Tag_Dict[self.Test_data[i-1]])
            else:
                Local_Tag_Dict=deepcopy(Temp_Tag_Dict)

            #print("i=",i)
            #print("Temp_Tag_Dict=",Temp_Tag_Dict)
            #print('Local_Tag_Dict=',Local_Tag_Dict)
            for ti_1 in Local_Tag_Dict:
                if (mu[(ti_1,i-1)]=='e'):
                    continue

                for ti in self.Tag_Dict[self.Test_data[i]]:
                    

                    if( (ti,ti_1) in self.P_Change):
                        try:
                            temp_p=math.log(self.P_Change[ (ti,ti_1)]*self.P_Emission[(self.Test_data[i],ti)])  ### log version
                            
                        except ValueError:
                            print("Math range error, Maybe you did not smooth or using wrong python interpreter!")
                            print("Did you set Addlambda(0)?")
                            print("Are you using python3 or python?")

                            sys.exit(0)

                        #temp_p=self.P_Change[ (ti,ti_1)]*self.P_Emission[(self.Test_data[i],ti)]  ### Probability version
                        temp_mu=mu[(ti_1,i-1)]+temp_p

                        #temp_mu=mu[ti_1,i-1]*temp_p ### probability version

                        if (mu[(ti,i)]=='e' or temp_mu > mu[(ti,i)]):
                            mu[(ti,i)]=temp_mu
                            backpointer[(ti,i)]=ti_1

            #### Prune the Tag_Dict after every i
            Temp_Tag_Dict=self.Prune(Pruning_rate,mu,self.Tag_Dict[self.Test_data[i]],i)
            

        #print("Loop ended")
        self.Result_Tag=[0 for w in range(0,L_test)]

        self.Result_Tag[-1]='###'
        for i in range(L_test-2,0,-1):
            self.Result_Tag[i]=backpointer[(self.Result_Tag[i+1],i+1)]

        self.Result_Tag[0]='###'

        
        S=mu[('###',L_test-1) ] 

        self.Accuracy(S,'Viterbi decoding') 


    def Posterior(self,Pruning_Rate):

        ### L_raw contains start and end ###
        L_raw=len(self.Raw_data)
        L_Tags=len(self.Tags)


        alpha = {}
        beta  = {}

        self.P_new_Tag={}        #P(C)_i    = 
        self.P_new_Transfer={}   #P(C->H)_i =
                            #(ti,ti_1,i)
        Temp_Tag_Dict=[]
        
        for ti in self.Tags:
            for i in range(0,L_raw):
                alpha[ (ti,i)] = 'e' ### this means 'minus infinity'
                beta [ (ti,i)] = 'e'

        alpha[('###',0)]= math.log(1) # log version
        

        for i in range(1,L_raw):
            
            #if(i%1000 ==0):
                #print("calculating alpha %d / %d " %(i,L_raw))
            
            if (not Temp_Tag_Dict):### if the initial case or no Pruning case
                                   ### Use the original Tag_Dict
                Local_Tag_Dict=deepcopy(self.Tag_Dict[self.Raw_data[i-1]])
            else:
                Local_Tag_Dict=deepcopy(Temp_Tag_Dict)


            for ti_1 in Local_Tag_Dict:
                
                for ti in self.Tag_Dict[self.Raw_data[i]]:
                    
                    if( (ti,ti_1) in self.P_Change):
                        log_temp_p=math.log(self.P_Change[ (ti,ti_1)]*self.P_Emission[(self.Raw_data[i],ti)]) 

                        alpha_p= alpha[ (ti_1,i-1)] + log_temp_p 

                        if (alpha[(ti,i)]=='e'):
                            alpha[(ti,i)] = alpha_p
                        else:
                            alpha[(ti,i)] = logaddexp(alpha[(ti,i)],alpha_p)

            Temp_Tag_Dict=self.Prune(Pruning_Rate,alpha,self.Tag_Dict[self.Raw_data[i]],i)
            #if(i!=L_raw-1):
            #    print('alpha=',math.exp(alpha[( 'C',i) ]))
        S=alpha[('###',L_raw-1)]
        #print("S=",math.exp(S))

        beta[('###',L_raw-1)] = math.log(1)
        
        ### i will not cover 0
        ### we must add P_new_Tag[('###',0)]
        self.P_new_Tag[('###',0)]=1

        Temp_Tag_Dict=[]
        for i in range(L_raw-1,0,-1):
            
            #if((L_raw-1-i)%1000 ==0):
                #print("calculating beta %d / %d " %( (L_raw-1-i),L_raw) )
            
            if (not Temp_Tag_Dict):### if the initial case or no Pruning case
                                   ### Use the original Tag_Dict
                Local_Tag_Dict=deepcopy(self.Tag_Dict[self.Raw_data[i]])
            else:
                Local_Tag_Dict=deepcopy(Temp_Tag_Dict)

            for ti in Local_Tag_Dict:

                self.P_new_Tag[ (ti,i) ]=math.exp(alpha[(ti,i)]+beta[(ti,i)]-S)

                for ti_1 in self.Tag_Dict[self.Raw_data[i-1]]:
                    
                    if( (ti,ti_1) in self.P_Change):
                        log_temp_p=math.log(self.P_Change[ (ti,ti_1)]*self.P_Emission[(self.Raw_data[i],ti)]) 

                        beta_p= beta[ (ti,i)] + log_temp_p 

                        if (beta[(ti_1,i-1)]=='e'):
                            beta[(ti_1,i-1)] = beta_p
                        else:
                            beta[(ti_1,i-1)] = logaddexp(beta[(ti_1,i-1)],beta_p)

                    self.P_new_Transfer[(ti,ti_1,i)]=math.exp(alpha[ (ti_1,i-1)]+log_temp_p+beta[(ti,i)]-S)
            
            Temp_Tag_Dict=self.Prune(Pruning_Rate,beta,self.Tag_Dict[self.Raw_data[i-1]],i-1)

        
        return S


    def EM(self,N,Smoother,Pruning_Rate,Only_Viterbi):

        L_raw=len(self.Raw_data)

        for Iter in range(0,N):
            

            if (Iter>0):
                self.OCS_iter=1

            if len(Smoother)==2:
                self.Add_lambda_Smooth(Smoother[1])
            else:
                self.One_count_Smooth()
            ### Use viterbi to do the tagging
            

            '''
            for key in self.P_Emission:
                print('Pwt[',key,']=',self.P_Emission[key])


            for key in self.P_Change:
                print('Ptt[',key,']=',self.P_Change[key])
            '''

            self.Viterbi(Pruning_Rate)
    
            ### Use F-B to update the counts
            ### Return S= log(P)
            
            if(not Only_Viterbi):
                S=self.Posterior(Pruning_Rate)

                Raw_Perplexity=math.exp (-S/(L_raw-1))
                print("Iteration ",Iter,": Perplexity per untagged raw word:",Raw_Perplexity)
                #print("S=",S)
                ### Re-estimate counts

                Total_Tag={}
                ### We should always add the train data to the counts
                self.Cwt=deepcopy(self.Cwt_Train)
                self.Ctt=deepcopy(self.Ctt_Train)

                for i in range(0,L_raw):
                    for tag in self.Tag_Dict[self.Raw_data[i]]:
                        Total_Tag[tag]=Total_Tag.get(tag, 0)+self.P_new_Tag.get((tag,i),0)

                        self.Cwt[tag]=self.Cwt.get(tag, 0)+self.P_new_Tag.get((tag,i),0) 
                        
                        self.Cwt[(self.Raw_data[i],tag)]=self.Cwt.get((self.Raw_data[i],tag),0)+self.P_new_Tag.get((tag,i),0)
                        if(i!=0):

                            self.Ctt[tag]=self.Ctt.get(tag, 0)+self.P_new_Tag.get((tag,i),0)
                            for tag_1 in self.Tag_Dict[ self.Raw_data[i-1] ]:
                                self.Ctt[(tag,tag_1)]=self.Ctt.get((tag,tag_1),0)+self.P_new_Transfer.get((tag,tag_1,i),0)

                
                #self.Add_lambda_Smooth(Smoother[1])



    def Check_Emission(self):

        print()
        print("Emission Possibility sum")
        ### Check if the emission sum is 1
        for tag in self.Tags:
            Tsum=0
            for data in self.Vocab:
                if(data,tag) in self.P_Emission:
                    Tsum+=self.P_Emission[(data,tag)]

            print(tag,Tsum)

    def Check_Change(self):
        print()
        print("change Possibility sum")
        for old_tag in self.Tags:
            Tsum=0
            for new_tag in self.Tags:
                if( (new_tag,old_tag) not in self.P_Change):
                    continue

                Tsum+=self.P_Change[(new_tag,old_tag)]

            print(old_tag,Tsum)

    def Accuracy(self,logP_Str,Decoder_Name):

        L_test=len(self.Test_data)

        Correct_known=0.0
        Correct_novel=0.0
        Correct_seen=0.0

        Sum_known=0.0
        Sum_seen=0.0
        Sum_novel=0.0
        
        for i in range(0,L_test):
            if self.Test_tag[i]=='###':
                continue
            else:
                if(self.Test_original_data[i] in self.Known_dict):
                    Sum_known+=1.0
                    
                    if self.Result_Tag[i]==self.Test_tag[i]:
                        Correct_known+=1.0
                

                elif(self.Test_original_data[i] in self.Seen_dict):
                    Sum_seen+=1.0
                    
                    if self.Result_Tag[i]==self.Test_tag[i]:
                        Correct_seen+=1.0
                
                elif(self.Test_original_data[i] in self.Novel_dict):
                    Sum_novel+=1.0
                    
                    if self.Result_Tag[i]==self.Test_tag[i]:
                        Correct_novel+=1.0

        Perplexity=math.exp (-logP_Str/(L_test-1.0))

        Accuracy_known = Correct_known / Sum_known
        
        if(Sum_seen!=0):
            Accuracy_seen = Correct_seen / Sum_seen
        else:
            Accuracy_seen = 0.0

        
        if(Sum_novel!=0):
            Accuracy_novel = Correct_novel / Sum_novel
        else:
            Accuracy_novel = 0.0

        Accuracy = (Correct_novel+Correct_seen+Correct_known)/(Sum_known+Sum_seen+Sum_novel)

        
        
        #if(Decoder_Name=='Viterbi decoding'):
        print("Tagging accuracy (%s):%.2f%% (known: %.2f%% seen: %.2f%% novel: %.2f%%)" % ( Decoder_Name,Accuracy*100, Accuracy_known*100,Accuracy_seen*100, Accuracy_novel*100 ))
        
        
        print("Perplexity per", Decoder_Name, "test word:", Perplexity)

if __name__=='__main__':
    
    try:
        Train_file=sys.argv[1]
        Test_file =sys.argv[2]
        Raw_file=sys.argv[3]
    except ValueError:
        sys.exit(0)

    P=Tagger( )

    print("Inilized")

    P.Read_Train(Train_file)
    print("Train Read")

    P.Read_Raw(Raw_file)
    print("Raw  Read")
    
    P.Read_Test_oov(Test_file)
    print("Test Read")
    

    ####### Parameter selection for this code!

    ### Em iterations!
    #EM_Iteration=1

    ### Smoother control
    #Smoother=('Add',1) ### Add lambda smoother
    Smoother= ['One_Count']  ### One-count smoother

    ### Pruning Rate
    Pruning_Rate=0.05

    ### If we only want to run Viterbi
    #Only_Viterbi=1; EM_Iteration=1 # If we only want to run Viterbi
                                   # We must set EM_Iteration=1
    Only_Viterbi=0; EM_Iteration=5


    
    P.EM(EM_Iteration,Smoother,Pruning_Rate,Only_Viterbi)


