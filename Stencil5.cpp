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

    partition_data(){}

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


    double& operator[](std::size_t idx) { return data_[idx]; }
    double operator[](std::size_t idx) const { return data_[idx]; }

    std::size_t size() const { return size_; }

private:
    std::shared_ptr<double[]> data_;
    std::size_t size_;

    friend class hpx::serialization::access;

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version) const {}
};

// The component  partion_server will allows for it to be created 
// and accessed remotely through a global address (hpx::id_type)
struct partition_server
  : hpx::components::component_base<partition_server>
{

public:

    // construct new instances
    partition_server() {}

    partition_server(partition_data const& data)
      : data_(data)
    {}

    partition_server(std::size_t size, double initial_value)
      : data_(size, initial_value)
    {}

    // access data
    partition_data get_data() const
    {
        return data_;
    }

    // Every member function which has to be invoked remotely needs to be
    // wrapped into a component action. The macro below defines a new type
    // 'get_data_action' which represents the (possibly remote) member function
    // partition::get_data().
    HPX_DEFINE_COMPONENT_DIRECT_ACTION(partition_server, get_data, get_data_action);

private:
    partition_data data_;
};

// Register the component

typedef hpx::components::component<partition_server> partition_server_type;
HPX_REGISTER_COMPONENT(partition_server_type, partition_server);

// Register the component action

typedef partition_server::get_data_action get_data_action;
HPX_REGISTER_ACTION(get_data_action);

// Wrtie the client side of the partition

struct partition : hpx::components::client_base<partition, partition_server>
{
    typedef hpx::components::client_base<partition, partition_server> base_type;

    partition() {}

    // Create new component on locality 'where' and initialize the held data
    partition(hpx::id_type where, std::size_t size, double initial_value)
      : base_type(hpx::new_<partition_server>(where, size, initial_value))
    {}

    // Create a new component on the locality co-located to the id 'where'. The
    // new instance will be initialized from the given partition_data.
    partition(hpx::id_type where, partition_data && data)
      : base_type(hpx::new_<partition_server>(hpx::colocated(where), std::move(data)))
    {}

    // Attach a future representing a (possibly remote) partition.
    partition(hpx::future<hpx::id_type> && id)
      : base_type(std::move(id))
    {}

    // Unwrap a future<partition> (a partition already holds a future to the id of the
    // referenced object, thus unwrapping accesses this inner future).
    partition(hpx::future<partition> && c)
      : base_type(std::move(c))
    {}

    ///////////////////////////////////////////////////////////////////////////
    // Invoke the (remote) member function which gives us access to the data.
    // This is a pure helper function hiding the async.
    hpx::future<partition_data> get_data() const
    {
        return hpx::async(get_data_action(), get_id());
    }
};


// Method to save the out to a file for ploting the result 
void saveFile(std::vector<partition> U){

  std::ofstream myfile;
  myfile.open("out.txt");
     
  for( size_t j = 0 ; j < U.size(); j++){

    for (size_t i = 0 ; i <  U[j].get_data().get().size(); i++)
          myfile <<   U[j].get_data().get()[i] << std::endl;
      
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

struct stepper
{
    // Our data for one time step
    typedef std::vector<partition> space;

    // Our operator
    static double heat(double left, double middle, double right)
    {
        return middle + (k*dt/(dx*dx)) * (left - 2*middle + right);
    }

    // The partitioned operator, it invokes the heat operator above on all elements
    // of a partition.
    static partition_data heat_part_data(partition_data const& left,
        partition_data const& middle, partition_data const& right)
    {
        // create new partition_data instance for next time step
        std::size_t size = middle.size();
        partition_data next(size);

        next[0] = heat(left[size-1], middle[0], middle[1]);

        for (std::size_t i = 1; i != size-1; ++i)
            next[i] = heat(middle[i-1], middle[i], middle[i+1]);

        next[size-1] = heat(middle[size-2], middle[size-1], right[0]);

        return next;
    }

    static partition heat_part(partition const& left, partition const& middle,
        partition const& right)
    {
        using hpx::dataflow;
        using hpx::util::unwrapping;

        return dataflow(
            unwrapping(
                [middle](partition_data const& l, partition_data const& m,
                    partition_data const& r)
                {
                    // The new partition_data will be allocated on the same
                    // locality as 'middle'.
                    return partition(middle.get_id(), heat_part_data(l, m, r));
                }
            ),
            left.get_data(), middle.get_data(), right.get_data());
    }

    // do all the work on 'np' partitions, 'nx' data points each, for 'nt'
    // time steps
    space do_work(std::size_t np, std::size_t nx, std::size_t nt);
};

// Register the plain action
HPX_PLAIN_ACTION(stepper::heat_part, heat_part_action);

stepper::space stepper::do_work(std::size_t np, std::size_t nx, std::size_t nt)
{
    using hpx::dataflow;

    // U[t][i] is the state of position i at time t.
    std::vector<space> U(2);
    for (space& s: U)
        s.resize(np);

    // Initial conditions: f(0, i) = i
    for (std::size_t i = 0; i != np; ++i)
        U[0][i] = partition(hpx::find_here(), nx, double(i));

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;
    using hpx::util::placeholders::_3;
    auto Op = hpx::util::bind(heat_part_action(), hpx::find_here(), _1, _2, _3);

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
    }

    // Return the solution at time-step 'nt'.
    return U[nt % 2];
}


int main(){
    
   stepper step;

   std::uint64_t t = hpx::util::high_resolution_clock::now();

   // Execute nt time steps on nx grid points and print the final solution.
   stepper::space solution = step.do_work(np, nx, nt);

   std::uint64_t elapsed = hpx::util::high_resolution_clock::now() - t;
   std::uint64_t const os_thread_count = hpx::get_os_thread_count();

   std::cout << "Computation took " << t / 1e9 << " on " << os_thread_count << " threads" << std::endl;
    
   std::cout << "Output written to out.txt!" << std::endl;

   saveFile(solution);

   return EXIT_SUCCESS;
}
