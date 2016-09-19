package barnes.dl4j;

import java.util.ArrayList;
import java.util.List;


public class Data {

    private final List<Integer> numbers;
    
    private Data(int ... n) {
        numbers = new ArrayList<>();
        for (int i : n) {
            getNumbers().add(i);
        }
    }
         
    public static Data of(int ... n) {
        return new Data(n);
    }

    public List<Integer> getNumbers() {
        return numbers;
    }
}
