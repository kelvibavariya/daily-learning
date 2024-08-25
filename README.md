# Date: 10-08-2024
1. DSA Practice
   1. Question: [House Robber](https://leetcode.com/problems/house-robber/)  
      Solution:
       * Memoization:
         ```cpp
         int solve(int i,vector<int>& nums,vector<int> &dp){
              if(i>=nums.size())return 0;
              if(dp[i]!=-1)return dp[i];
              int take=nums[i]+solve(i+2,nums,dp); //can not rob adjacent house
              int not_take=solve(i+1,nums,dp);
              return dp[i]=max(take,not_take);
          }
          int rob(vector<int>& nums) {
              vector<int> dp(nums.size(),-1);
              return solve(0,nums,dp);
          }
          ```
         Complexity:  
           -> TC: O(N)  
           -> SC: O(N) + O(N) - dp arr + stack space  
       * Tabulation:
           ```cpp
           int rob(vector<int>& nums) {
             int n=nums.size();
             if(n==0)return 0;
             if(n==1)return nums[0];
    
             vector<int> dp(n,0);
             dp[0]=nums[0];
             dp[1]=max(nums[0],nums[1]);
            
             for(int i=2;i<n;i++){
                dp[i]=max(dp[i-1],nums[i]+dp[i-2]);
             }
             return dp[n-1];
           }
           ```
           Complexity:  
             -> TC: O(N)  
             -> SC: O(N)
    2. Question: [Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)  
       Solution:  
       ```cpp
       int maxProfit(vector<int>& prices) {
         int min_price=INT_MAX; //to buy
          int profit=0; //max profit
          for(int i=0;i<prices.size();i++)
          {
              if (prices[i]<min_price)
                  min_price=prices[i];
              else if(prices[i]-min_price>profit)
                  profit=prices[i]-min_price;
          }
          return profit;
       }
       ```
       Complexity:  
         -> TC: O(N)  
         -> SC: O(1)
     3. Question: [Best Time to Buy and Sell stock with Transaction Fee](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)  
        Solution:
        * Memoization:
          ```cpp
          int solve(int i,int buy,vector<int>& prices, int fee,vector<vector<int>> &dp){
            if(i>=prices.size())return 0;
            if(dp[i][buy]!=-1)return dp[i][buy];
            //pay transaction fee while selling
            if(buy){
                return dp[i][buy]=max(prices[i]-fee+solve(i+1,0,prices,fee,dp),solve(i+1,buy,prices,fee,dp));
            }
            else{
                return dp[i][buy]=max(-prices[i]+solve(i+1,1,prices,fee,dp),solve(i+1,buy,prices,fee,dp));
            }
          }
          int maxProfit(vector<int>& prices, int fee) {
            vector<vector<int>> dp(prices.size(),vector<int>(2,-1));
            //buy-false -> can only buy
            //buy-true ->  can only sell
            return solve(0,0,prices,fee,dp);
          }
          ```
          Complexity:  
           -> TC: O(N * 2)  
           -> SC: O(N * 2) + O(N) - 2D dp arr + recursion stack space  
        * Tabulation:
          ```cpp
          int maxProfit(vector<int>& prices, int fee) {
            int n=prices.size();
            vector<vector<int>> dp(n+1,vector<int>(2,0));
            for(int i=n-1;i>=0;i--){
                for(int j=0;j<=1;j++){
                    //can only buy
                    if(j==0){
                        dp[i][j]=max(-prices[i]+dp[i+1][1],dp[i+1][j]);
                    }
                    //can only sell
                    else{
                        dp[i][j]=max(prices[i]-fee+dp[i+1][0],dp[i+1][j]);
                    }
                }
            }
            return dp[0][0];
          }
          ```
          Complexity:  
           -> TC: O(N * 2)  
           -> SC: O(N * 2)
    4. Question: [Best Time to Buy and Sell stock with Cooldown](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)  
       Solution:
       ```cpp
       int maxProfit(vector<int>& prices) {
        int n=prices.size();
        vector<vector<int>> dp(n+2,vector<int>(2,0));
        int profit;
        for(int i=n-1;i>=0;i--){
           for(int j=0;j<=1;j++){
               if(j){
                    profit=max(prices[i]+dp[i+2][0],dp[i+1][1]);
                }
                else{
                    profit=max(-prices[i]+dp[i+1][1],dp[i+1][0]);
                }
                dp[i][j]=profit;
           } 
        }
        return dp[0][1];
       }
       ```
       Complexity:  
        -> TC: O(N * 2)  
        -> SC: O(N * 2)
   5. Question: [Best Time to Buy and Sell Stock iii](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/)  
      Solution:
      ```cpp
      int maxProfit(vector<int>& p) {
        int n=p.size();
        vector<vector<vector<int>>> dp(n+1,vector<vector<int>>(2,vector<int>(3,0)));
        for(int i=n-1;i>=0;i--){
            for(int j=0;j<=1;j++){
                for(int k=1;k<=2;k++){
                    if(j)
                        dp[i][j][k]=max(p[i]+dp[i+1][0][k-1],dp[i+1][1][k]);
                    else
                        dp[i][j][k]=max(-p[i]+dp[i+1][1][k],dp[i+1][0][k]);
                }
            }
        }
        return dp[0][0][2];
      }
      ```
      Complexity:  
        -> TC: O(N * 2 * 3)  
        -> SC: O(N * 2 * 3)
# Date: 11-08-2024   
1. DSA Practice
   1. Question: [Best Time to Buy and Sell Stock iv](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/)  
      Solution:
      ```cpp
      int maxProfit(int k, vector<int>& prices) {
       int n = prices.size();
       vector<vector<vector<int>>> dp(n + 1, vector<vector<int>>(2, vector<int>(k + 1, 0)));
   
       for (int i = n - 1; i >= 0; i--) {
           for (int buy = 0; buy <= 1; buy++) {
               for (int j = 1; j <= k; j++) {
                   if (buy == 0) { // can buy only
                       dp[i][buy][j] = max(dp[i + 1][0][j], -prices[i] + dp[i + 1][1][j]);
                   }
   
                   else { // can sell only
                       dp[i][buy][j] = max(dp[i + 1][1][j], prices[i] + dp[i + 1][0][j - 1]);
                   }
               }
           }
       }
   
       return dp[0][0][k];
      }
      ```  
      Complexity:  
      -> TC: O(N * 2 * K)  
      -> SC: O(N * 2 * K)

# Date: 12-08-2024
1. DSA Practice
   1. Question: [Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/)  
      Solution:  
      ```cpp
      string longestPalindrome(string s) {
        if (s.length() <= 1) {
            return s;
        }
        
        int max_len = 1;
        int start = 0;
        int end = 0;
        vector<vector<bool>> dp(s.length(), vector<bool>(s.length(), false));
        
        for (int i = 0; i < s.length(); ++i) {
            dp[i][i] = true;
            for (int j = 0; j < i; ++j) {
                // (i-j<=2) - string of length <= 3 - always palindrome
                // (j [(j+1)...(i-1)] i) - s[j]==s[i] && middle is also palindrome 
                if (s[j] == s[i] && (i - j <= 2 || dp[j + 1][i - 1])) {
                    dp[j][i] = true;
                    if (i - j + 1 > max_len) {
                        max_len = i - j + 1;
                        start = j;
                        end = i;
                    }
                }
            }
        }
        
        return s.substr(start, end - start + 1);
      }
      ```
      Complexity:  
      -> TC: O(N * N)  
      -> SC: O(N * N)
   2. Question: [Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/)  
      Solution:
      ```cpp
      //memoization
      int solve(int i,int j,string &s1,string &s2,vector<vector<int>> &dp){
        if(i>=s1.length() || j>=s2.length())return 0;
        if(dp[i][j]!=-1)return dp[i][j];
        if(s1[i]==s2[j])return dp[i][j]=1+solve(i+1,j+1,s1,s2,dp);
        return dp[i][j]=max(solve(i,j+1,s1,s2,dp),solve(i+1,j,s1,s2,dp));
       }
      int longestCommonSubsequence(string text1, string text2) {
           int n=text1.length(),m=text2.length();
           vector<vector<int>> dp(n+1,vector<int>(m+1,0));
           // tabulation
           for(int i=n-1;i>=0;i--){
               for(int j=m-1;j>=0;j--){
                   if(text1[i]==text2[j])
                       dp[i][j]=1+dp[i+1][j+1];
                   else dp[i][j]=max(dp[i][j+1],dp[i+1][j]);
               }
           }
           return dp[0][0];
      }
      ```  
      Complexity:  
      -> TC: O(N * M)  
      -> SC: O(N * M)

# Date: 13-08-2024
1. DSA practice
   1. Question: [Longest Palindromic Subsequence](https://leetcode.com/problems/longest-palindromic-subsequence/)    
      Solution:
      ```cpp
      // Memoization
      int solve(int i,int j,string s,vector<vector<int>> &dp){
        if(i>j)return 0;
        if(i==j)return 1;
        if(dp[i][j]!=-1)return dp[i][j];
        if(s[i]==s[j])
            return dp[i][j]=2+solve(i+1,j-1,s,dp);
        else
        return dp[i][j]=max(solve(i+1,j,s,dp),solve(i,j-1,s,dp));
      }
      int longestPalindromeSubseq(string s) {
        int n=s.length();
        vector<vector<int>> dp(n,vector<int>(n,0));
        // Tabulation
        for (int i = 0; i < n; i++) {
            dp[i][i] = 1;
        }
        for (int i = n - 2; i >= 0; i--) { // i goes from n-2 down to 0
            for (int j = i + 1; j < n; j++) { // j goes from i+1 up to n-1
        // for(int j=1;j<n;j++){ //end of substring
        //     for(int i=j-1;i>=0;i--){ //start of substring
                if(s[i]==s[j]){
                    dp[i][j]=2+dp[i+1][j-1];
                }
                else{
                    dp[i][j]=max(dp[i+1][j],dp[i][j-1]);
                }
            }
        }
        return dp[0][n-1];
        // return solve(0,n-1,s,dp);
      }
      ```  
      Note: another solution find longest common subsequence of (s, reverse(s))  
      Complexity:  
      -> TC: O(N * N)  
      -> SC: O(N * N)

# Date: 14-08-2024  
1. DSA Practice
   1. Question: [Edit Distance](https://leetcode.com/problems/edit-distance/)  
      Solution:
      ```cpp
      // memoization
      int solve(int i,int j,string word1, string word2, vector<vector<int>> &dp){
        if(i>=word1.size())return word2.length()-j; //insert remaining
        if(j>=word2.size())return word1.length()-i; //delete remaining
        
        if(dp[i][j]!=-1)return dp[i][j];
        if(word1[i]==word2[j])
            return dp[i][j]=solve(i+1,j+1,word1,word2,dp);
        else
            //(i,j+1) -> insert into s
            //(i+1,j) -> delete from s
            //(i+1,j+1) -> replace in s
            return dp[i][j]=1+min(solve(i,j+1,word1,word2,dp),
            min(solve(i+1,j,word1,word2,dp), solve(i+1,j+1,word1,word2,dp)));
      }
      // tabulation
      int minDistance(string word1, string word2) {
        int n=word1.size(),m=word2.size();
        vector<vector<int>> dp(n+1,vector<int>(m+1,0));
        //base case
        for(int i=0;i<n;i++)dp[i][m]=n-i;
        for(int i=0;i<m;i++)dp[n][i]=m-i;
        
        for(int i=n-1;i>=0;i--){
            for(int j=m-1;j>=0;j--){
                if(word1[i]==word2[j])
                    dp[i][j]=dp[i+1][j+1];
                else
                    dp[i][j]=1+min(dp[i][j+1],min(dp[i+1][j],dp[i+1][j+1]));
            }
        }
        return dp[0][0];
        // return solve(0,0,word1,word2,dp);
      }
      ```
      Complexity:  
      -> TC: O(N * M)  
      -> TC: O(N * M)

# Date: 15-08-2024
1. DSA Practice
   1. Question: [Minimum ascii delete sum for two strings](https://leetcode.com/problems/minimum-ascii-delete-sum-for-two-strings/)  
      Solution:
      ```cpp
      // memoization
      int solve(int i,int j,string s1, string s2, vector<vector<int>> &dp){
        if(i>=s1.length() && j>=s2.length())return 0;
        if(i>=s1.length()){
            int sum=0;
            for(int p=j;p<s2.length();p++){
                sum+=s2[p]-'a'+'a';
            }
            return sum;
        }
        if(j>=s2.length()){
            int sum=0;
            for(int p=i;p<s1.length();p++){
                sum+=s1[p]-'a'+'a';
            }
            return sum;
        }
        if(dp[i][j]!=-1)return dp[i][j];
        if(s1[i]==s2[j])return dp[i][j]=solve(i+1,j+1,s1,s2,dp);
        else return dp[i][j]=min(s1[i]-'a'+'a'+solve(i+1,j,s1,s2,dp), s2[j]- 
           'a'+'a'+solve(i,j+1,s1,s2,dp));
      }
      // tabulation
      int minimumDeleteSum(string s1, string s2) {
        int n=s1.length(),m=s2.length();
        vector<vector<int>> dp(n+1,vector<int>(m+1,0));
        //base case condition to notice
        for(int i=n-1;i>=0;i--){
            dp[i][m]=dp[i+1][m]+s1[i]-'a'+'a';
        }
        for(int i=m-1;i>=0;i--){
            dp[n][i]=dp[n][i+1]+s2[i]-'a'+'a';
        }
        for(int i=n-1;i>=0;i--){
            for(int j=m-1;j>=0;j--){
                if(s1[i]==s2[j])
                    dp[i][j]=dp[i+1][j+1];
                else
                    dp[i][j]=min(s1[i]-'a'+'a'+dp[i+1][j], s2[j]-'a'+'a'+dp[i][j+1]);
            }
        }
        return dp[0][0];
        // return solve(0,0,s1,s2,dp);
      }
      ```  
      Complexity:  
      -> TC: O(N * M)  
      -> SC: O(N * M)  

# Date: 16-08-2024
1. DSA Practice
   1. Question: [Shortest Common Supersequence](https://leetcode.com/problems/shortest-common-supersequence/)  
      Solution:
      ```cpp
      string shortestCommonSupersequence(string str1, string str2) {
        int n=str1.size(),m=str2.size();
        vector<vector<int>> dp(n+1,vector<int>(m+1,0));
        for(int i=n-1;i>=0;i--){
            for(int j=m-1;j>=0;j--){
                if(str1[i]==str2[j])dp[i][j]=1+dp[i+1][j+1];
                else dp[i][j]=max(dp[i+1][j],dp[i][j+1]);
            }
        }
        string ans="";
	    // start with (0,0) because answer is stored at dp[0][0]
        int i=0,j=0;
        while(i<n && j<m){
		  // lcs characters
            if(str1[i]==str2[j]){
                ans+=str1[i];
                i++;
                j++;
            }
		  // non-lcs characters
            else if(dp[i][j+1]>dp[i+1][j]){
                ans+=str2[j];
                j++;
            }
            else{
                ans+=str1[i];
                i++;
            }
        }
        while(i<n){
            ans+=str1[i];
            i++;
        }
        while(j<m){
            ans+=str2[j];
            j++;
        }
        return ans;
      }
      ```
      Complexity:  
      -> TC: O(N * M)  
      -> SC: O(N * M)  
  2. Question: [Distinct Subsequences](https://leetcode.com/problems/distinct-subsequences/)  
     Solution:  
     ```cpp
     int solve(int i,int j,string s,string t,vector<vector<int>> &dp){
        if(j>=t.length())return 1;
        if(i>=s.length())return 0;
        if(dp[i][j]!=-1)return dp[i][j];
        if(s[i]==t[j])return dp[i][j]=solve(i+1,j+1,s,t,dp)+solve(i+1,j,s,t,dp);
        else return dp[i][j]=solve(i+1,j,s,t,dp);
     }
     int numDistinct(string s, string t) {
        int n=s.length(),m=t.length();
        vector<vector<double>> dp(n+1,vector<double>(m+1,0));
        for(int i=0;i<=n;i++)dp[i][m]=1;
        for(int i=n-1;i>=0;i--){
            for(int j=m-1;j>=0;j--){
                if(s[i]==t[j])
                    dp[i][j]=dp[i+1][j+1]+dp[i+1][j];
                else
                    dp[i][j]=dp[i+1][j];
            }
        }
        return dp[0][0];
        // return solve(0,0,s,t,dp);
     }
     ```
     Complexity:  
     -> TC: O(N * M)  
     -> SC: O(N * M)  

# Date: 17-08-2024  
1. DSA Practice
   1. Question: [Unique Binary Search Trees](https://leetcode.com/problems/unique-binary-search-trees/)  
      Solution:  
      ```cpp
      int numTrees(int n) {
        vector<int> v(n+1);
        v[0]=1;v[1]=1;
        for(int i=2;i<=n;i++){ // i as a root for bst of [1 to n]
            for(int j=1;j<=i;j++){ // j as a root for bst of [1 to i]
                v[i]+=v[j-1]*v[i-j];
            }
        }
        return v[n];
      }
      ```
      Complexity:  
      -> TC: O(N * N)  
      -> SC: O(N)  

# Date: 18-08-2024
1. DSA Practice
   1. Question: [Burst Balloons](https://leetcode.com/problems/burst-balloons/)  
      Solution:  
      ```cpp
      // memoization
      int solve(int i,int j,vector<int>& nums, vector<vector<int>> &dp){
        if(i>j)return 0;
        if(dp[i][j]!=-1)return dp[i][j];
        int ans=INT_MIN;
        // burst kth balloon at the end in i to j - so kth adjacent ballons are i-1 and j+1
        for(int k=i;k<=j;k++){
            int temp=nums[i-1]*nums[k]*nums[j+1]+solve(i,k-1,nums,dp)+solve(k+1,j,nums,dp);
            ans=max(ans,temp);
        }
        return dp[i][j]=ans;
      }
      // tabulation
      int maxCoins(vector<int>& nums) {
        int n=nums.size();
        nums.insert(nums.begin(),1);
        nums.push_back(1);
        vector<vector<int>> dp(n+2,vector<int>(n+2,0));
        for(int i=n;i>=1;i--){
            for(int j=1;j<=n;j++){
                if(i>j)continue;
                int ans=INT_MIN;
                for(int k=i;k<=j;k++){
                    int temp=nums[i-1]*nums[k]*nums[j+1]+dp[i][k-1]+dp[k+1][j];
                    ans=max(ans,temp);
                }
                dp[i][j]=ans;
            }
        }
        return dp[1][n];
        // return solve(1,n,nums,dp);
      }
      ```
      Complexity:   
      -> TC: O(N * N * N)  
      -> SC: O(N * N)  

# Date: 19-08-2024  
1. DSA Practice
   1. Question: [2 keys keyboard](https://leetcode.com/problems/2-keys-keyboard/)  
      Solution:
      ```cpp
      int solve(int onTheScreen,int copied,int n){
        if(onTheScreen>=n)return 0;
	     // copy and paste
        if(n%onTheScreen==0)return 2+solve(onTheScreen+onTheScreen,onTheScreen,n);
	     // paste
        else return 1+solve(onTheScreen+copied,copied,n);
      }
      int minSteps(int n) {
        vector<vector<int>> dp(n + 1, vector<int>(n + 1, INT_MAX));

       // Base case - when onTheScreen = n
       for(int copied = 0; copied <= n; copied++) {
           dp[n][copied] = 0;
       }
   
       for(int onTheScreen = n - 1; onTheScreen >= 1; onTheScreen--) {
           for(int copied = onTheScreen; copied >= 0; copied--) {
               if (n % onTheScreen == 0 && onTheScreen + onTheScreen <= n && dp[onTheScreen + onTheScreen][onTheScreen] != INT_MAX) {
                   dp[onTheScreen][copied] = min(dp[onTheScreen][copied], 2 + dp[onTheScreen + onTheScreen][onTheScreen]);
               }
               if (onTheScreen + copied <= n && dp[onTheScreen + copied][copied] != INT_MAX ) {
                   dp[onTheScreen][copied] = min(dp[onTheScreen][copied], 1 + dp[onTheScreen + copied][copied]);
               }
           }
       }
           return dp[1][0];
           // return solve(1,0,n);
      }
      ```
      Complexity:  
      -> TC: O(N * N)  
      -> SC: O(N * N)    

# Date: 20-08-2024  
1. DSA Practice
   1. Question: [Minimum cost tree from leaf values](https://leetcode.com/problems/minimum-cost-tree-from-leaf-values/)  
      Solution:
      ```cpp
      int solve(int i,int j,vector<int>& arr,map<pair<int,int>,int> mp,vector<vector<int>> &dp){
        if(i>=j)return 0;
        if(dp[i][j]!=-1)return dp[i][j];
        int ans=INT_MAX;
        for(int k=i;k<j;k++){
            ans=min(ans,mp[{i,k}]*mp[{k+1,j}]+solve(i,k,arr,mp,dp)+solve(k+1,j,arr,mp,dp));
        }
        return dp[i][j]=ans;
      }
      int mctFromLeafValues(vector<int>& arr) {
        int n=arr.size();
        map<pair<int,int>,int> mp;
        for(int i=0;i<n;i++){
            mp[{i,i}]=arr[i];
            for(int j=i+1;j<n;j++){
                mp[{i,j}]=max(mp[{i,j-1}],arr[j]);
            }
        }
        vector<vector<int>> dp(n+1,vector<int>(n+1,-1));
        return solve(0,n-1,arr,mp,dp);
      }
      ```
      Complexity:  
      -> TC: O(N * N)  
      -> SC: O(N * N)  

# Date: 21-08-2024  
1. DSA Practice
   1. Question: [Guess number higher or lower II](https://leetcode.com/problems/guess-number-higher-or-lower-ii/)  
      Solution:  
      ```cpp
      int solve(int i,int j,vector<vector<int>> &dp){
        if(i>=j)return 0;
        if(dp[i][j]!=-1)return dp[i][j];  
        int ans=INT_MAX;
        for(int k=i;k<=j;k++){ // guess every number from [i,j]
            int temp=k+max(solve(i,k-1,dp),solve(k+1,j,dp)); //max from lower/higher (max loss)
            ans=min(ans,temp); //minimize maximum loss
        }
        return dp[i][j]=ans;
      }
      int getMoneyAmount(int n) {
        vector<vector<int>> dp(n+1,vector<int>(n+1,0));
        for(int i=n;i>=1;i--){ //i-> 1 to n
            for(int j=i+1;j<=n;j++){ //j-> n to i-1 (base case: i>=j)
                if(i == j){ // base case
                    dp[i][j]=0;
                    continue;
                }
                int ans=INT_MAX;
                for(int k=i;k<=j;k++){
                    int temp=k+max(solve(i,k-1,dp),solve(k+1,j,dp));
                    ans=min(ans,temp);
                }
                dp[i][j]=ans;
            }
        }
        return dp[1][n];
        // return solve(1,n,dp);
      }
      ```
      Complexity:  
      -> TC: O(N * N * N)  
      -> SC: O(N * N)  
