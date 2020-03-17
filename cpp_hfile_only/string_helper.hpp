#include <iostream>
#include <string>
#include <functional>

namespace sth {

	/*
	*Return splited string
	*/
	std::vector<std::string> split(std::string st, char sp) {
		std::vector<std::string> ret;
		std::string::size_type pos1, pos2;
		pos2 = st.find(sp);
		pos1 = 0;
		while (std::string::npos != pos2)
		{
			ret.push_back(st.substr(pos1, pos2 - pos1));

			pos1 = pos2 + 1;
			pos2 = st.find(sp, pos1);
		}
		ret.push_back(st.substr(pos1));
		return ret;
	}

	/*
	* Check if string had any part is palindrome
	*/
	bool is_palindrome(std::string st,size_t min_lenth=3) {
		std::size_t re=0;
		std::hash<std::string> ha;
		for (size_t i = 0; i < st.length() - min_lenth; i++)
		{
			re = i;
			while (re!= std::string::npos)
			{
				re = st.find(st[i], re + 1);
				if (re == std::string::npos)break;
				if ((re - i + 1) < min_lenth)continue;
				if (ha(std::string(st.begin() + re, st.end())) == \
					ha(std::string(st.rbegin(), st.rend() - re)))return true;
			}
		}
		return false;
	}
}