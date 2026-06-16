using UnityEngine;
using UnityEngine.InputSystem;
using Unity.MLAgents.Input;

public class SimpleMovement : MonoBehaviour, IInputActionAssetProvider
{
    private float speed = 5f;
    private PlayerInput playerInput;
    private InputAction moveAction;
    private Vector2 currentInput;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        playerInput = GetComponent<PlayerInput>();
        
        moveAction = playerInput.actions["Move"];
    }

    // Update is called once per frame
    void Update()
    {
        currentInput = moveAction.ReadValue<Vector2>();

        transform.Translate(new Vector3(currentInput.x, currentInput.y, 0) * speed * Time.deltaTime);
    }

    public (InputActionAsset, IInputActionCollection2) GetInputActionAsset()
    {
        if (playerInput == null)
        {
            playerInput = GetComponent<PlayerInput>();
        }
        return (playerInput.actions, playerInput.actions);
    }
}
