
/* set history of all species at initial time to 0  */



void init_history(int trial)  {
    int ncell,n_gene;
    for (ncell=0;ncell<NCELLTOT;ncell++){
    	for (n_gene=0;n_gene<SIZE;n_gene++){
      	    history[n_gene][0][ncell]=0;
        }
    }



}
