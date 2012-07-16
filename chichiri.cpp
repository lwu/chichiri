//
// Chichiri -- an implementation of agglomerative \chi^2 data discretization for data mining
//
// Leslie Wu
//

#include <iostream>
#include <fstream>
#include <algorithm>
#include <utility>
#include <vector>
#include <set>
#include <map>
#include <list>
#include <string>
#include <cassert>
#include <sstream>
#include <iterator>

using namespace std;

// Tuple vector
typedef float DataType;
typedef string ClassType;

typedef vector<DataType> FloatVec;
typedef pair<FloatVec, ClassType> Tuple;
typedef vector<Tuple> TupleVec;

TupleVec g_data;
set<ClassType> g_classTypes;
map<ClassType, int> g_globalClassCount;

// Intervals (an interval corresponds to a set of indices)
typedef int TupleIndex;
typedef set<TupleIndex> IntervalSet;
typedef list<IntervalSet> IntervalList;

IntervalList g_intervals;

// Constants
#define NUM_CLASSES 4 // hardcoded for simplicity
int g_max_intervals = 6;

// logged output
ostringstream olog;

// Print command-line usage information
void usage()
{
	cerr << "\n" "usage:"
		 << "\n" "\t chichiri input.dat [max intervals]" << endl;
}

// Return true iff no non-whitespace characters between here and EOL
bool iseol(istream& is)
{
	// Eat whitespace
	while (!is.eof() && isspace(is.peek())) {		
		if (is.get() == '\n') return true;
	}
	
	return false;
}

// read data from input file [infile], store data in
//  [g_data] and create [g_classTypes] vector
void read_data(ifstream& infile)
{
	// hardcode data format
	g_data.clear();

	while (!infile.eof()) {
		char ch = infile.peek();
		
		if (!isdigit(ch)) {
			return;
		}

		vector<float> f(NUM_CLASSES);
		char dummy;
		string class_type;

		// read NUM_FLOATS floats per line
		for (int i=0; i < NUM_CLASSES; i++) {
			infile >> f[i] >> dummy;
			cout << f[i] << ", ";
		}
		
		// eat rest of line
		getline(infile, class_type);

		// insert into list (set) of class types
		g_classTypes.insert(class_type);

		cout << class_type;

		cout << endl;

		g_data.push_back(make_pair(f, class_type));
	}
}

// Debugging -- print one interval
void print_interval_set(IntervalSet& indices)
{
	copy(indices.begin(), indices.end(), ostream_iterator<int>(cout, ", "));
	cout << endl;
}

// Debugging -- print all intervals
void print_all_intervals()
{
	IntervalList::iterator lit = g_intervals.begin(),
		lend = g_intervals.end();

	cout << "[intervals]" << endl;

	for ( ; lit != lend; ++lit) {
		IntervalSet& indices = *lit;
		print_interval_set(indices);
	}
}

// Initialize intervals, one for each unique attribute value
void initialize_intervals(int dimIndex)
{
	g_intervals.clear();

	IntervalSet interval;

	TupleVec::iterator it = g_data.begin(), end = g_data.end(), next;
	int index = 0;
	for ( ; it != end; ++it, ++index) {
		next = it+1;

		// Add element (index) to interval
		interval.insert(index);

		bool insertInterval = (next == end) ||
			( (next->first)[dimIndex] != (it->first)[dimIndex] );

		if (insertInterval) {
			// Insert interval into list if
			//  (a) end of sequence or
			//  (b) next element is not the same as current element
			g_intervals.push_back(interval);
			interval.clear();
		}
	}

	// Debug
	print_all_intervals();
}

// Count how many instances of each class
void count_classes()
{
	TupleVec::iterator it = g_data.begin(), end = g_data.end();
	for ( ; it != end; ++it) {
		ClassType classType = it->second;
		g_globalClassCount[classType]++;
	}
}

// Compute $\chi^2$ value for an interval
float chisquared_interval(IntervalSet& interval)
{
	// Count instances of each class
	map<ClassType, int> classCount;

	IntervalSet::iterator it = interval.begin(),
		end = interval.end();	
	for ( ; it != end; ++it) {
		int index = *it;
        ClassType classType = g_data[index].second;
		classCount[classType]++;
	}

	// Keep track of summation
	float chichiri = 0.0f;

	set<ClassType>::iterator sit = g_classTypes.begin(),
		send = g_classTypes.end();

	for ( ; sit != send; ++sit) {
		ClassType classType = *sit;

		// Uses notation from section 4.2.1,
		// "Discretization: An Enabling Technique" by H. Liu et al.

		float A_ij = static_cast<float>(classCount[classType]);

		float R_i  = static_cast<float>(interval.size());
		float C_j  = static_cast<float>(g_globalClassCount[classType]);
		float N    = static_cast<float>(g_data.size());

		float E_ij = (R_i * C_j) / N;

		float top = (A_ij - E_ij);

		chichiri += top*top / E_ij;
	}

	cout << "{" << chichiri << "} + ";

	return chichiri;
}

// Compute $\chi^2$ for adjacent intervals
float compute_chisquared(IntervalSet& interval_1, 
						 IntervalSet& interval_2)
{
    return 
		chisquared_interval(interval_1) +
		chisquared_interval(interval_2);
}

// Templatized dimensionally-indexed tuple comparison operator
template <class Tuple>
struct tuple_less_than
{
	tuple_less_than(int d) : dim(d) { }

	bool operator()(const Tuple& lhs, const Tuple& rhs) {
		return (lhs.first[dim] < rhs.first[dim]);
	}

	int dim;
};

// Find adjacent intervals with smallest $\chi^2$
IntervalList::iterator find_min_chi_chi()
{
	// Compute $\chi^2$ value for each adjacent interval,
	//  keeping track of minimum value
	IntervalList::iterator lit = g_intervals.begin(),
		lend = g_intervals.end();
	IntervalList::iterator next = lit, min_lit = lend;

	float min_chisquared = 1e6;
	bool first = true;

	for ( ; lit != lend; ++lit) {
        next = lit;
		++next;

		if (next == lend) break;

		float chisquared = compute_chisquared(*lit, *next);

		// Debug
		print_interval_set(*lit);
		cout << "\\chi^2 = " << chisquared << endl;

		if (first || chisquared < min_chisquared) {
			min_lit = lit;
			min_chisquared = chisquared;
		}

		first = false;
	}

	cout << "min_chisquared = " << min_chisquared << endl;

    return min_lit;
}

pair<DataType, DataType> get_range(IntervalSet& interval, int dimIndex)
{
	assert(!interval.empty());

	int beginIndex = *interval.begin();
	IntervalSet::iterator end = interval.end();
	--end;
	int endIndex = *end;

	assert(beginIndex >= 0 && beginIndex < (int)g_data.size());
	assert(endIndex >= 0 && endIndex < (int)g_data.size());

	return make_pair(
		g_data[beginIndex].first[dimIndex], 
		g_data[endIndex].first[dimIndex]);
}

void print_interval_summary(ostream& os, int dimIndex)
{
	IntervalList::iterator lit = g_intervals.begin(),
		lend = g_intervals.end();
	IntervalList::iterator next = lit;

	vector<DataType> split_points;

	os << "\n" "Feature " << (dimIndex+1) << ":" << endl;
	os << "Ranges: ";

	for ( ; lit != lend; ++lit) {
        next = lit;
		++next;

		pair<DataType, DataType> range, next_range;
		range = get_range(*lit, dimIndex);

		if (next != lend) {
            next_range = get_range(*next, dimIndex);

			float average = (range.second + next_range.first) * 0.5f;
			split_points.push_back(average);
		}

		os << "[" << range.first << ", " << range.second << "] ";
	}

	os << endl;

	os << "Split points: ";
	copy(split_points.begin(), split_points.end(),
		ostream_iterator<DataType>(os, ", "));
	os << endl;
}

// $\chi^2$ just for one dimension
void chi_chi_dim_analysis(int dimIndex)
{	
	// Sort and initialize one interval per unique attribute value

	sort(g_data.begin(), g_data.end(), tuple_less_than<Tuple>(dimIndex));

	TupleVec::iterator tit = g_data.begin(), tend = g_data.end();

	cout << "[sort]" << endl;

	int index = 0;
	for ( ; tit != tend; ++tit, ++index) {
		cout << index << ":";
		copy(tit->first.begin(), tit->first.end(), ostream_iterator<float>(cout, ", "));
		cout << tit->second << endl;		
	}

	initialize_intervals(dimIndex);

	// Count instances of all classes
	count_classes();

	while ((int)g_intervals.size() > g_max_intervals) {
		// Find adjacent intervals with smallest $\chi^2$
		IntervalList::iterator min_lit = find_min_chi_chi();
		assert(min_lit != g_intervals.end());

		IntervalList::iterator min_lit_next = min_lit;
		++min_lit_next;

		cout << "[before merge] ";
		print_all_intervals();

		// Merge
		IntervalSet& interval_1 = *min_lit;
		IntervalSet& interval_2 = *min_lit_next;

		interval_1.insert(interval_2.begin(), interval_2.end());

		g_intervals.erase(min_lit_next);

		cout << "[after merge] ";
		print_all_intervals();
	}

	// Debugging
	print_interval_summary(cout, dimIndex);

	// Logged output
	print_interval_summary(olog, dimIndex);
}

// Perform $\chi^2$ analysis
void chi_chi_analysis()
{
	// Run $\chi^2$ analysis once for each dimension
	for (int dimIndex=0; dimIndex < NUM_CLASSES; dimIndex++) {
		chi_chi_dim_analysis(dimIndex);
	}
}

int main(int argc, char* argv[])
{
	if (argc < 2) {
		usage();
		return 1;
	}

	// Read data file
	char* filename = argv[1];

	cout << "\n" "Reading [" << filename << "]..." << endl;

	ifstream infile(filename);

	if (!infile) {
		cerr << "Couldn't read [" << filename << "]" << endl;
		return 1;
	}

	// Second optional argument is # intervals
	if (argc >= 3) {
		const char* max_intervals_str = argv[2];
        g_max_intervals = atoi(max_intervals_str);

		g_max_intervals = max(1, g_max_intervals); // sanity check
	}

	cout << "max intervals = " << g_max_intervals << endl;

	read_data(infile);

	// Process
	chi_chi_analysis();

	// Results
	cout << "\n\n" 
		"Results:\n" 
		"--------" << endl;
	cout << olog.str();

	return 0;
}
