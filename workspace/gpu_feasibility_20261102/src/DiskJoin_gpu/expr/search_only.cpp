// Search-only driver: runs DiskJoin::search() on FROZEN cluster/metadata/hnsw
// artifacts without executing one_level_kmeans build (which would overwrite
// the frozen files). Reads-only; instrumentation env vars still supported.
#include "../utils/utils.h"
#include "../lib/DiskJoin.h"
#include <iostream>

int main(int argc, char** argv) {
    ConfigReader config_reader(argv[1]);
    Timer timer;
    DiskJoin index;
    timer.tick();
    index.search(config_reader);
    timer.tuck("Join done");
}
