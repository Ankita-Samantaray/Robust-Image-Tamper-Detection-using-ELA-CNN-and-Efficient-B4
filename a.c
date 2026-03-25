
#include<bits/stdc++.h>
using namespace std;
int getSum(int n) {
    if(n == 0) return n;
    else{
    return n + getSum(n - 1);  
    }  
}
int main()
{
    int n;
    cout << "Enter a number : "; 
    cin >> n;
    
    int sum = getSum(n);
    
    cout << sum;
    
    return 0;
}