#include <hpx/hpx.hpp>
#include <hpx/include/parallel_algorithm.hpp>
#include<vector>
#include<fstream>
#include <iostream>
#include <memory>
#include <utility>
#include <boost/range/irange.hpp>
#include<hpx/hpx_main.hpp>

// New class to manage the partitions
class partition_data
{
public:
    partition_data(std::size_t size)
      : data_(new double[size]), size_(size)
    {}

    partition_data(std::size_t size, double initial_value)
      : data_(new double[size]),
        size_(size)
    {
        double base_value = double(initial_value * size);
        for (std::size_t i = 0; i != size; ++i)
            data_[i] = base_value + double(i);
    }

    partition_data(partition_data && other)
      : data_(std::move(other.data_))
      , size_(other.size_)
    {}

    double& operator[](std::size_t idx) { return data_[idx]; }
    double operator[](std::size_t idx) const { return data_[idx]; }

    std::size_t size() const { return size_; }

private:
    std::unique_ptr<double[]> data_;
    std::size_t size_;
};

// Method to save the out to a file for ploting the result 
void saveFile(std::vector<hpx::shared_future<partition_data>> U){

  std::ofstream myfile;
  myfile.open("out.txt");
     
  for( size_t j = 0 ; j < U.size(); j++){
    for (size_t i = 0 ; i <  U[j].get().size(); i++)
          myfile <<   U[j].get()[i] << std::endl;
      
  }
  myfile.close();

}

// Function to map the indicies
 
inline std::size_t idx(std::size_t i, int dir, std::size_t size)
{
    if(i == 0 && dir == -1)
        return size-1;
    if(i == size-1 && dir == +1)
        return 0;

    HPX_ASSERT((i + dir) < size);

    return i + dir;
}

// Default parameters

double k = 0.5;     // heat transfer coefficient
double dt = 1.;     // time step
double dx = 1.;     // grid spacing
size_t nx = 100;
size_t nt = 45;
size_t np = 1;
size_t nd = 5; // maximal depth of the tree

// Simulation class

class stepper
{

public: 

    // Our data for one time step
    typedef hpx::shared_future<partition_data> partition;
    typedef std::vector<partition> space;

    // Our operator
    static double heat(double left, double middle, double right)
    {
        return middle + (k*dt/(dx*dx)) * (left - 2*middle + right);
    }

    // The partitioned operator, it invokes the heat operator above on all
    // elements of a partition.
    static partition_data heat_part(partition_data const& left,
        partition_data const& middle, partition_data const& right)
    {
        std::size_t size = middle.size();
        partition_data next(size);

        next[0] = heat(left[size-1], middle[0], middle[1]);

        for(std::size_t i = 1; i != size-1; ++i)
        {
            next[i] = heat(middle[i-1], middle[i], middle[i+1]);
        }

        next[size-1] = heat(middle[size-2], middle[size-1], right[0]);

        return next;
    }

    // do all the work on 'np' partitions, 'nx' data points each, for 'nt'
    // time steps, limit depth of dependency tree to 'nd'
    hpx::future<space> do_work(std::size_t np, std::size_t nx, std::size_t nt,
        std::uint64_t nd)
    {
        using hpx::dataflow;
        using hpx::util::unwrapping;

        // U[t][i] is the state of position i at time t.
        std::vector<space> U(2);
        for (space& s: U)
            s.resize(np);

        // Initial conditions: f(0, i) = i
        std::size_t b = 0;
        auto range = boost::irange(b, np);
        using hpx::parallel::execution::par;
        hpx::parallel::for_each(par, std::begin(range), std::end(range),
            [&U, nx](std::size_t i)
            {
                U[0][i] = hpx::make_ready_future(partition_data(nx, double(i)));
            }
        );

        // limit depth of dependency tree
        hpx::lcos::local::sliding_semaphore sem(nd);

        auto Op = unwrapping(&stepper::heat_part);

        // Actual time step loop
        for (std::size_t t = 0; t != nt; ++t)
        {
            space const& current = U[t % 2];
            space& next = U[(t + 1) % 2];

            for (std::size_t i = 0; i != np; ++i)
            {
                next[i] = dataflow(
                        hpx::launch::async, Op,
                        current[idx(i, -1, np)], current[i], current[idx(i, +1, np)]
                    );

            }

            // every nd time steps, attach additional continuation which will
            // trigger the semaphore once computation has reached this point
            if ((t % nd) == 0)
            {
                next[0].then(
                    [&sem, t](partition &&)
                    {
                        // inform semaphore about new lower limit
                        sem.signal(t);
                    });
            }

            // suspend if the tree has become too deep, the continuation above
            // will resume this thread once the computation has caught up
            sem.wait(t);
        }

        // Return the solution at time-step 'nt'.
        return hpx::when_all(U[nt % 2]);
    }
};

int main(){
    
   stepper step;

   std::uint64_t t = hpx::util::high_resolution_clock::now();

   // Execute nt time steps on nx grid points and print the final solution.
   hpx::future<stepper::space> result = step.do_work(np, nx, nt, nd);

   stepper::space solution = result.get();

   hpx::wait_all(solution);



   std::uint64_t elapsed = hpx::util::high_resolution_clock::now() - t;
   std::uint64_t const os_thread_count = hpx::get_os_thread_count();

   std::cout << "Computation took " << t / 1e9 << " on " << os_thread_count << " threads" << std::endl;
    
   std::cout << "Output written to out.txt!" << std::endl;

   saveFile(solution);

   return EXIT_SUCCESS;
}
