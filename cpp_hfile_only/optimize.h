#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <math.h>
#include <functional>

/*
	For 01question | DynamicPlanning
*/
template <typename DataType, typename TargetType>
std::vector<DataType> SingleTargetApproximate(std::vector<DataType> datas, TargetType target) {

	DataType sum = 0;
	bool _global = false;
	for (auto& x : datas) { sum += x; }
	if (sum < target)return datas;

	std::sort(datas.begin(), datas.end());
	std::vector<DataType> retlist;
	std::function<TargetType(std::vector<DataType> datas,
		TargetType target,
		std::vector<DataType> & _ret)> subF;
	subF = [&_global, &subF](
		std::vector<DataType> datas, 
		TargetType target,
		std::vector<DataType>& _ret
		) {
		// Hit
		if (_global) { return 0; }
		if (target <= 0) {
			_global = true; return 0;
		}
		// For only one
		else if (datas.size() <= 1) {
			if (target< datas[0]) {
				return target;	
			}
			else
			{
				_ret.push_back(datas[0]);
				if ((target - datas[0]) == 0)_global = true;
				return (target - datas[0]);
			}
		}
		// Normal
		else {
			DataType _dend,tmp,minVal;
			std::size_t p_minVal;

			std::vector<TargetType> rtValue;
			std::vector<DataType> rtTmp;
			std::vector<std::vector<DataType>> rtList;
			// Generate all sublist recursively
			for (size_t i = datas.size(); i > 0; i--)
			{
				if (target < datas[i - 1])continue;
				tmp = datas[i - 1];
				datas.erase(datas.begin()+i - 1);
				rtValue.push_back(subF(datas, target - tmp, rtTmp));
				
				if (_global) { 
					for(size_t i=0;i< rtTmp.size();i++)_ret.push_back(rtTmp[i]);
					_ret.push_back(tmp);
					return 0; 
				}
				datas.push_back(tmp);
				std::sort(datas.begin(), datas.end());
				rtTmp.push_back(tmp);
				rtList.push_back(rtTmp);
				rtTmp.clear();
			}
			if (_global) { return 0; }
			minVal = rtValue[0]; p_minVal = 0;
			// Choose minimum det
			for (size_t i = 1;i < rtValue.size(); ++i)
			{
				if (rtValue[i] < minVal) {
					minVal = rtValue[i];
					p_minVal = i;
				}
			}
			for(size_t i=0;i< (rtList[p_minVal]).size();i++)_ret.push_back((rtList[p_minVal])[i]);
			return minVal;
		}
	};
	subF(datas, target, retlist);
	return retlist;
}



/*
	For 01question | DynamicPlanning
	PS: This function is VALUATION, NOT exact solution
*/
template <typename DataType, typename TargetType>
std::vector<DataType> ValuationSingleTargetApproximate(std::vector<DataType> datas, TargetType target) {
	std::vector<DataType> retlist = {};
	std::sort(datas.begin(), datas.end());
	DataType sum=0,minDet=0,det;
	int minDetMem=-1;
	for (auto& x : datas) { sum += x; }
	if (sum < target)return datas;
	det=minDet = sum / datas.size();
	while (datas.size()>0 && target>= datas[0])
	{	
		if (minDetMem!=-1 && target >= datas[minDetMem]) {
			target -= datas[minDetMem];
			sum -= datas[minDetMem];
			retlist.push_back(datas[minDetMem]);
			datas.erase(datas.begin() + minDetMem);
		}
		for (auto i = 0; i < datas.size(); i++)
		{
			if (datas[i] > target) { 
				sum -= datas[i];
				datas.erase(datas.begin()+i);
			}
		}
		minDetMem = -1;
		det = minDet = datas.size()?(sum / datas.size()):sum;
		for (auto i = 0; i < datas.size(); i++)
		{
			if (abs(datas[i] - det) < minDet) {
				minDetMem = i;
				minDet = abs(datas[i] - det);
			}
		}
	}
	return retlist;
}