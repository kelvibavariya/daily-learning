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
   
