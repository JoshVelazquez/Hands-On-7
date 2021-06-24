package examples.backpropagation;

import jade.core.Agent;
import jade.core.behaviours.*;

public class BackpropagationAgent extends Agent {
    private Backpropagation backpropagation;

    protected void setup() {
        backpropagation = new Backpropagation();
        addBehaviour(new entrenar());
    }

    private class entrenar extends OneShotBehaviour {

        public void action() {
            System.out.println("Agent's action method executed");
            backpropagation.entrenar();
        }

        public int onEnd() {
            myAgent.doDelete();
            return super.onEnd();
        }
    }
}
